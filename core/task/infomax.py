from typing import Optional, Tuple
import pytorch_lightning as pl

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from core.models.mi_estimator import MutualInformationEstimator


class InfoMax(pl.LightningModule):
    def __init__(
            self,
            mi_estimator: MutualInformationEstimator,
            encoder_x: Optional[nn.Module] = None,
            encoder_y: Optional[nn.Module] = None,
            same_encoder: bool = False,
    ):
        super(InfoMax, self).__init__()

        self.mi_estimator = mi_estimator
        self.encoder_x = encoder_x
        assert not same_encoder or encoder_y is not None, "If same_encoder is False, encoder_y must be None"
        if same_encoder:
            encoder_y = encoder_x
        self.encoder_y = encoder_y

        print(self)

    def on_fit_start(self) -> None:
        # Set the entropy of a and y if the computation is specified in the dataloader
        if hasattr(self.trainer.datamodule, "h_a"):
            self.mi_estimator.h_a = self.trainer.datamodule.h_a
        if hasattr(self.trainer.datamodule, "h_y"):
            self.mi_estimator.h_y = self.trainer.datamodule.h_y

        # Set the number of negative samples consistently to the number specified on the datamodule
        if hasattr(self.trainer.datamodule, "neg_samples"):
            if self.mi_estimator.neg_samples != self.trainer.datamodule.neg_samples:
                print(f"Warning: The number of negative samples specified in the datamodule ({self.trainer.datamodule.neg_samples}) does not match the number of negative samples specified in the estimator ({self.mi_estimator.neg_samples}).")
                print(f"Setting the number of negative samples in the data_loader to {self.mi_estimator.neg_samples}")
                self.trainer.datamodule.neg_samples = self.mi_estimator.neg_samples

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None,
            a: Optional[torch.Tensor] = None,
            step: str = 'train'
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Compute a lower bound for I(x,y).
        Args:
            x: a tensor with shape [N, D] in which x[i] is sampled from p(x)
            y: a tensor with shape [N, D] or [N, M, D] in which y[i,j] is sampled from p(y|x[i])
            y_: a tensor with shape [N, D] or [N, M', D] in which y[i,j] is sampled from r(y|x[i])
            a: a tensor with shape [N, D] representing the "attributes" corresponding to x
            step: the step of the training (train, val, test)
        Returns:
            mi_value, mi_grad: A tuple consisting of 1) the estimation for I(x,y) and 2) a quantity to differentiate to
                maximize mutual information. Note that 1) and 2) can have different values.
        """

        if y.ndim == x.ndim:
            # If one dimension is missing, we assume there is only one positive sample
            y = y.unsqueeze(1)

        assert y.ndim == x.ndim + 1

        if y_ is not None:
            if y_.ndim == x.ndim:
                # If one dimension is missing, we assume there is only one negative sample
                y_ = y_.unsqueeze(1)
            assert y_.ndim == x.ndim + 1

        # Compute the ratio using the primal bound
        primal_value, primal_grad = self.mi_estimator.compute_primal_ratio(x, y, a)

        # And the rest using the dual density ratio
        dual_value, dual_grad = self.mi_estimator.compute_dual_ratio(x, y, y_)

        mi_grad = primal_grad + dual_grad

        if primal_value is not None:

            mi_value = primal_value + dual_value

            self.log(f"I_pr(x;y)/{step}/value", primal_value, on_step=True, on_epoch=True)
            self.log(f"I(x;y)/{step}/value", mi_value, on_step=True, on_epoch=True, prog_bar=True)
        else:
            mi_value = None

        self.log(f"KL_f(p||r)/{step}/value", dual_value, on_step=True, on_epoch=True)
        self.log(f"KL_f(p||r)/{step}/grad", dual_grad, on_step=True, on_epoch=True)
        self.log(f"I_pr(x;y)/{step}/grad", primal_grad, on_step=True, on_epoch=True)
        self.log(f"I(x;y)/{step}/grad", mi_grad, on_step=True, on_epoch=True, prog_bar=True)

        return mi_value, mi_grad

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x = batch['x']
        y = batch['y']

        if 'y_' in batch:
            y_ = batch['y_']
            a = batch['a']
        else:
            y_ = None
            a = None

        if self.encoder_x is not None:
            x = self.encoder_x(x)
        if self.encoder_y is not None:
            y = self.encoder_y(y)
            y_ = self.encoder_y(y_)

        mi_value, mi_grad = self(x, y, y_, a, step='train')

        return {"loss": -mi_grad, "value": mi_value, "x": x, "y": y, "y_": y_, "a": a}
