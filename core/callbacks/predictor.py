from typing import Optional, Any, Dict, Type, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.distributions import Categorical
from pyro.nn import DenseNN

from pytorch_lightning import Callback, Trainer, LightningModule

from core.models.predictor import ConditionalCategoricalMLP


class PredictorCallback(Callback):
    def __init__(
            self,
            input_dim: int,
            n_classes: int,
            hidden_dims: List[int],
            optimizer_class: Type[Optimizer] = Adam,
            optimizer_init_args: Optional[Dict[str, Any]] = None,
            train_on='z_y',
            gamma=0.9,
            t=1.0,
            tol=0.001
    ):
        super().__init__()
        if optimizer_init_args is None:
            optimizer_init_args = {'lr': 1e-3}

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.optimizer_class = optimizer_class
        self.optimizer_init_args = optimizer_init_args
        self.train_on = train_on
        self.gamma = gamma

        self.predictor_nn: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scale = None
        self.t = t
        self.t_anneal = 0.5
        self.t_min = 0.05
        self.tol = tol

        self._freeze = False
        self._recovered_callback_state: Optional[Dict[str, Any]] = None
        self._marginal = None
        self._stored_a = None
        self._mi = 0.0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.predictor_nn = DenseNN(
            input_dim=self.input_dim,
            param_dims=[self.n_classes],
            hidden_dims=self.hidden_dims
        ).to(
            pl_module.device
        )

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )

        if accel.is_distributed:
            raise NotImplementedError()

        self.optimizer = self.optimizer_class(self.predictor_nn.parameters(), **self.optimizer_init_args)

        if self._recovered_callback_state is not None:
            self.predictor_nn.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def compute_logits(self, y, update_estimation=False):
        logits = self.predictor_nn(y)
        current_scale = logits.std(0)

        if self.scale is None:
            scale = current_scale
        else:
            scale = (self.gamma * self.scale + (1 - self.gamma) * current_scale).detach()

        if update_estimation:
            self.scale = scale

        logits = logits / scale / self.t

        # Normalize the log probabilities
        logits = logits - torch.logsumexp(logits, -1).unsqueeze(-1)

        return logits

    def compute_mi(
        self,
        y: torch.Tensor,
        update_estimation: bool = False,
    ):
        logits = self.compute_logits(y, update_estimation=update_estimation)

        # Compute the marginal probabilities
        current_log_p_a = torch.logsumexp(logits, 0) - np.log(logits.shape[0])

        if self._marginal is None:
            log_p_a = current_log_p_a
        else:
            log_p_a = self.gamma * self._marginal + (1-self.gamma) * current_log_p_a

        if update_estimation:
            self._marginal = log_p_a.detach()

        # Compute the mutual information
        h_a_y = -(logits.exp() * logits).sum(-1).mean(0)
        h_a = -(current_log_p_a.exp() * log_p_a).sum(-1)

        mi = h_a - h_a_y

        return mi

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        assert self.train_on in outputs or self.train_on in batch, f"The outputs or batches must contain {self.train_on}"

        if self.train_on in outputs:
            y = outputs[self.train_on]
        else:
            y = batch[self.train_on]

        assert "idx" in batch, "The batches must contain idx"
        idx = batch["idx"]

        mi = self.compute_mi(y.detach(), update_estimation=not self._freeze)

        if not self._freeze:
            loss = -mi.mean()

            # update finetune weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        pl_module.log(f"I({self.train_on};a)/train", mi, on_step=True, on_epoch=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        assert self.train_on in outputs or self.train_on in batch, f"The outputs or batches must contain {self.train_on}"

        if self.train_on in outputs:
            y = outputs[self.train_on]
        else:
            y = batch[self.train_on]

        mi = self.compute_mi(y.detach(), update_estimation=False)

        self._mi = self.gamma * self._mi + (1-self.gamma) * mi.detach()
        pl_module.log(f"I({self.train_on};a)/val", mi, on_step=False, on_epoch=True)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print("t", self.t) # TODO: remove
        if self._mi - np.log(self.n_classes) < self.tol:
            self.t = max(self.t * self.t_anneal, self.t_min)
        else:
            self._freeze = True
            print("Done!")

    def state_dict(self) -> dict:
        return {"state_dict": self.predictor_nn.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._recovered_callback_state = state_dict







