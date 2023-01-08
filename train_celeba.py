from argparse import ArgumentParser
from functools import partial

import numpy as np
from torch import nn
import torch.nn.functional as F

from typing import Optional
from pyro.distributions import ConditionalDistribution
from pytorch_lightning import Trainer
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50

from core.callbacks.online_ssl import SSLOnlineEvaluator
from core.models.encoder import VisionTransformer


class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, no_batch_norm=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if no_batch_norm:
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(normalized_shape=self.hidden_dim, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim, bias=False),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim, bias=False),
            )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class AdaptedSimCLR(SimCLR):
    def __init__(self, *args, predictor: Optional[ConditionalDistribution] = None, no_batch_norm: bool = False, **kwargs):
        self.no_batch_norm = no_batch_norm
        super().__init__(*args, **kwargs)
        self.predictor = predictor
        self.projection = Projection(
            input_dim=self.hidden_mlp,
            hidden_dim=self.hidden_mlp,
            output_dim=self.feat_dim,
            no_batch_norm=self.no_batch_norm
        )
        print(self)

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50
        else:
            raise NotImplementedError(f"Architecture {self.arch} not supported")

        if self.no_batch_norm:
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        encoder = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=8,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=self.hidden_mlp,
            out_dim=self.hidden_mlp,
        )

        return encoder

    def forward(self, x):
        return self.encoder(x)

        # return backbone(
        #     first_conv=self.first_conv,
        #     maxpool1=self.maxpool1,
        #     return_all_feature_maps=False,
        #     norm_layer=norm_layer
        # )

    def shared_step(self, batch):
        # Adapted to deal with dicts
        img1 = batch['x']
        img2 = batch['y']

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        out = {"loss": loss, "z1": h1, "z2": h2}
        out["contrastive_loss"] = out["loss"]+0

        # Add prediction Loss
        if self.predictor is not None:
            a = batch["a"]
            q_a_Y = self.predictor.condition(h2)
            log_q_A_Y = q_a_Y.log_prob(a)
            rec_loss = -log_q_A_Y.mean(0)
            out["loss"] += rec_loss
            out["rec_loss"] = rec_loss

        return out

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)

        self.log("loss/train", output["loss"], on_step=True, on_epoch=True)
        self.log("loss/contrastive/train", output["contrastive_loss"], on_step=True, on_epoch=True)
        if "rec_loss" in output:
            self.log("loss/rec/train", output["rec_loss"], on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)

        self.log("loss/val", output["loss"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("loss/contrastive/val", output["contrastive_loss"], on_step=False, on_epoch=True)
        if "rec_loss" in output:
            self.log("loss/rec/val", output["rec_loss"], on_step=False, on_epoch=True)
        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = SimCLR.add_model_specific_args(parent_parser)
        parser.add_argument('--use_predictor', action='store_true')
        parser.add_argument('--sample_same_attributes', action='store_true')
        parser.add_argument('--no_batch_norm', action='store_true')
        parser.add_argument('--default_root_dir', type=str, default='.')

        return parser

from core.task import InfoMax
class EInfoMax(InfoMax):
    def configure_optimizers(self):
        import torch
        from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
        params = self.parameters()

        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-6)

        warmup_steps = 1000
        total_steps = 100000

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

def cli_main():
    from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelCheckpoint

    parser = ArgumentParser()

    # model args
    parser = AdaptedSimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    ###############
    # DataLoaders #
    ###############

    if args.dataset == 'celeba':
        from core.data.datamodule import CelebABatchDataModule

        args.hidden_mlp = 512
        train_attributes = np.arange(40)[1::4]
        # Entropy of the attribute distribution
        args.num_samples = 162770

        dm = CelebABatchDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_attributes=train_attributes,
            sample_same_attributes=args.sample_same_attributes,
        )

        args.input_height = 224
        args.jitter_strength = 0.5
        ssl_t_dim = 30
        ssl_num_classes = 2
        # TODO: add Normalization
        normalization = None
    else:
        raise NotImplemented()

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    #########
    # Model #
    #########
    if args.use_predictor:
        from core.models.predictor import ConditionalLinearCategorical
        predictor = ConditionalLinearCategorical(y_dim=args.hidden_mlp, n_classes=2, a_dim=10)
        args.predictor = predictor

    # model = AdaptedSimCLR(**args.__dict__)

    from core.models.encoder import VisionTransformer
    from core.models.mi_estimator import SimCLR

    model = EInfoMax(
        encoder_x=VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=8,
            num_heads=8,
            hidden_dim=256,
            mlp_dim=args.hidden_mlp,
            out_dim=args.hidden_mlp,
        ),
        mi_estimator=SimCLR(
            x_dim=args.hidden_mlp,
            y_dim=args.hidden_mlp,
            hidden_dims=[512],
            out_dim=128
        )
    )


    ###########
    # Trainer #
    ###########

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="loss/val")
    # online_evaluator = SSLOnlineEvaluator(z_dim=args.hidden_mlp, num_classes=ssl_num_classes, t_dim=ssl_t_dim)
    # online_evaluator
    callbacks = [model_checkpoint] if args.online_ft else [model_checkpoint]
    callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32, #if args.fp32 else 16,
        default_root_dir=args.default_root_dir,
        callbacks=callbacks,
        # fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()





