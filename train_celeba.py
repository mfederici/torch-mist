from argparse import ArgumentParser

import numpy as np

from typing import Optional
from pyro.distributions import ConditionalDistribution
from pytorch_lightning import Trainer
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR

class AdaptedSimCLR(SimCLR):
    def __init__(self, *args, predictor: Optional[ConditionalDistribution] = None, h_a: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = predictor
        self.h_a = h_a

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

        # Add prediction Loss
        if self.predictor is not None:
            a = batch["a"]
            q_a_Y = self.predictor.condition(h2)
            log_q_A_Y = q_a_Y.log_prob(a)
            loss += -log_q_A_Y.mean(0)
            if self.h_a is not None:
                loss += self.h_a

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = SimCLR.add_model_specific_args(parent_parser)
        parser.add_argument('--use_predictor', action='store_true')
        parser.add_argument('--h_a', type=float, default=None)
        parser.add_argument('--sample_same_attributes', action='store_true')

        return parser
def cli_main():
    from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform

    parser = ArgumentParser()

    # model args
    parser = AdaptedSimCLR.add_model_specific_args(parser)
    args = parser.parse_args()

    ###############
    # DataLoaders #
    ###############

    if args.dataset == 'celeba':
        from core.data.datamodule import CelebADataModule

        args.hidden_mlp = 512
        train_attributes = np.arange(40)[1::4]
        # Entropy of the attribute distribution
        args.h_a = 4.152420957448182
        args.num_samples = 162770

        dm = CelebADataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_attributes=train_attributes,
            sample_same_attributes=args.sample_same_attributes,
        )

        args.input_height = 218
        args.jitter_strength = 0.5
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
        from core.models.predictor import ConditionalCategoricalMLP
        predictor = ConditionalCategoricalMLP(y_dim=args.hidden_mlp, n_classes=2, hidden_dims=[512, 128], a_dim=10)
        args.predictor = predictor

    model = AdaptedSimCLR(**args.__dict__)

    ###########
    # Trainer #
    ###########

    # lr_monitor = LearningRateMonitor(logging_interval="step")
    # model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    # callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]
    # callbacks.append(lr_monitor)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        # callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()





