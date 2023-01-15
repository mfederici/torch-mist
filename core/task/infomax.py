from typing import Optional, Tuple, Dict, Union, Any, Type
import pytorch_lightning as pl

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from core.models.encoder import SlowlyUpdatingModel, EncoderKeywords
from core.models.mi_estimator import MutualInformationEstimator


class InfoMax(pl.LightningModule):
    def __init__(
            self,
            mi_estimator: MutualInformationEstimator,
            optimizer_class: Type[torch.optim.Optimizer],
            optimizer_init_args: Optional[Dict[str, Any]] = None,
            lr_scheduler_class: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
            lr_scheduler_init_args: Optional[Dict[str, Any]] = None,
            lr_scheduler_params: Optional[Dict[str, Any]] = None,
            encoder_x: Optional[nn.Module] = None,
            encoder_y: Optional[Union[EncoderKeywords]] = EncoderKeywords.same,
            tau: float = 0.99,
    ):
        """
        InfoMax model based on the specified mutual information estimator and (optionally) encoders.
        Args:
            mi_estimator: A mutual information estimator.
            encoder_x: An encoder for the input x (Optional).
            encoder_y: An encoder for the input y (Optional).
                        If "same", the encoder for x is used,
                        if "slow", a slowly updating encoder is used.
            tau: The temperature for the slowly updating encoder (used only if encoder_y="slow").
        """
        super(InfoMax, self).__init__()

        self.mi_estimator = mi_estimator
        self.encoder_x = encoder_x
        self.encoder_y_str = None
        if isinstance(encoder_y, EncoderKeywords):
            self.encoder_y_str = encoder_y
            if encoder_y == EncoderKeywords.same:
                encoder_y = encoder_x
            elif encoder_y == EncoderKeywords.slow:
                encoder_y = SlowlyUpdatingModel(encoder_x, tau)
            else:
                raise ValueError(f"Unknown value for encoder_y: {encoder_y}")

        self.encoder_y = encoder_y
        self.optimizer_class = optimizer_class
        self.optimizer_init_args = optimizer_init_args
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_init_args = lr_scheduler_init_args
        self.lr_scheduler_params = lr_scheduler_params

        print(self)

    def on_fit_start(self) -> None:
        # Set the entropy of a and y if the computation is specified in the dataloader
        if hasattr(self.trainer, "datamodule"):
            if hasattr(self.trainer.datamodule, "h_a"):
                self.mi_estimator.h_a = self.trainer.datamodule.h_a
            if hasattr(self.trainer.datamodule, "h_y"):
                self.mi_estimator.h_y = self.trainer.datamodule.h_y

            # Set the number of negative samples consistently to the number specified on the datamodule
            # This is necessary only when using a predictor
            predictor = None
            if hasattr(self.mi_estimator, "predictor"):
                predictor = self.mi_estimator.predictor
            if hasattr(self.trainer.datamodule, "neg_samples") and predictor is not None:
                if self.mi_estimator.neg_samples != self.trainer.datamodule.neg_samples:
                    print(
                        "Warning: The number of negative samples specified in the datamodule ("
                        f"{self.trainer.datamodule.neg_samples}) does not match the number of negative samples "
                        f"specified in the estimator ({self.mi_estimator.neg_samples}).")
                    print(
                        f"Setting the number of negative samples in the data_loader to {self.mi_estimator.neg_samples}")
                    self.trainer.datamodule.neg_samples = self.mi_estimator.neg_samples

    def forward(
            self,
            x: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.encoder_x is not None:
            x = self.encoder_x(x)
        return x

    def shared_step(self, batch: Dict[str, torch.Tensor], step: str) -> STEP_OUTPUT:
        x = batch['x']
        y = batch['y']

        if 'y_' in batch:
            y_ = batch['y_']
        else:
            y_ = None
        if 'a' in batch:
            a = batch['a']
        else:
            a = None

        if self.encoder_x is not None:
            x = self.encoder_x(x)
        if self.encoder_y is not None:
            y = self.encoder_y(y)
            if y_ is not None:
                y_ = self.encoder_y(y_)

        estimates = self.mi_estimator(x, y, y_, a)
        estimates["loss"] = -estimates["mi/grad"]

        if self.encoder_x:
            estimates["z_x"] = x
            if step == "val" or step == "test":
                # If the batch contains the original image o, encode it (for SSL validation).
                if "o" in batch:
                    with torch.no_grad():
                        estimates["z_o"] = self.encoder_x(batch["o"])
        if self.encoder_y:
            estimates["z_y"] = y

        return estimates

    def log_components(self, results: Dict[str, torch.Tensor], step: str):
        for name, value in results.items():
            if value.shape == torch.Size([]):
                self.log(f"{name}/{step}", value, on_step=step == "train", on_epoch=True)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        results = self.shared_step(batch, "train")
        self.log_components(results, "train")

        return results

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        results = self.shared_step(batch, "val")
        self.log_components(results, "val")
        return results

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_init_args)
        if self.lr_scheduler_class is not None:

            if "interval" in self.lr_scheduler_params:

                if self.lr_scheduler_params["interval"] == "step":
                    steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
                    for param in self.lr_scheduler_init_args:
                        if param.endswith("steps"):
                            self.lr_scheduler_init_args[param] *= steps_per_epoch

            scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_init_args)
            if self.lr_scheduler_params is not None:
                scheduler = {
                    'scheduler': scheduler,
                    **self.lr_scheduler_params
                }
            return [optimizer], [scheduler]
        else:
            return optimizer

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += "\n  (mi_estimator): " + self.mi_estimator.__repr__().replace('\n', '\n  ')
        if self.encoder_x:
            s += "\n  (encoder_x): " + self.encoder_x.__repr__().replace('\n', '\n  ')
        if self.encoder_y_str == EncoderKeywords.same and self.encoder_x is not None:
            s += f"\n  (encoder_y): same as encoder_x"
        elif isinstance(self.encoder_y, nn.Module):
            s += "\n  (encoder_y): " + self.encoder_y.__repr__().replace('\n', '\n  ')
        s += "\n)"
        return s
