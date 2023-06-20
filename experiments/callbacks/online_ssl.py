from typing import Optional, Dict, Any

import torch
from torch import Tensor
from torch.optim import Optimizer, Adam
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_warn

from miest.distributions.transforms import ConditionalDistributionModule
from miest.models.learnable_distributions import ConditionalCategoricalMLP


# Implementation based on pl_bolts.models.callbacks.ssl_online
class SSLOnlineEvaluator(Callback):
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    """

    def __init__(
        self,
        z_dim: int,
        train_on: str,
        evaluate_on: str,
        target: str = 't',
        hidden_dim: int = 256,
        num_classes: Optional[int] = None,
        t_dim: int = 1,
        dataset: Optional[str] = None,
    ):
        """
        Args:
            z_dim: Representation dimension
            hidden_dim: Hidden dimension for the fine-tuned MLP
        """
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.t_dim = t_dim
        self.train_on = train_on
        self.evaluate_on = evaluate_on
        self.target = target

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[ConditionalDistributionModule] = None
        self.num_classes: Optional[int] = num_classes
        self.dataset: Optional[str] = dataset

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = ConditionalCategoricalMLP(
            y_dim=self.z_dim,
            n_classes=self.num_classes,
            hidden_dims=[self.hidden_dim],
            a_dim=self.t_dim
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
            if accel.use_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            elif accel.use_dp:
                from torch.nn.parallel import DataParallel as DP

                self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

        self.optimizer = Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def shared_step(
        self,
        z: Tensor,
        t: Tensor,
    ):

        q_t_Z = self.online_evaluator.condition(z.float().detach())
        log_q_T_Z = q_t_Z.log_prob(t)
        loss = -log_q_T_Z.mean()
        return loss

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        assert self.train_on in outputs, f"The outputs must contain {self.train_on}"
        assert self.target in batch, f"The batch must contain the target {self.target}"

        z = outputs[self.train_on]
        t = batch["t"]

        mlp_loss = self.shared_step(z, t)

        # update finetune weights
        self.optimizer.zero_grad()
        mlp_loss.backward()
        self.optimizer.step()

        pl_module.log(f"log q({self.target}|{self.train_on})/train", -mlp_loss, on_step=True, on_epoch=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        assert self.evaluate_on in outputs, f"The outputs must contain {self.evaluate_on}"
        assert self.target in batch, f"The batch must contain the target {self.target}"

        z = outputs[self.evaluate_on]
        t = batch[self.target]

        mlp_loss = self.shared_step(z, t)

        pl_module.log(f"log q({self.target}|{self.evaluate_on})/val", -mlp_loss, on_step=False, on_epoch=True)

    def state_dict(self) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._recovered_callback_state = state_dict

