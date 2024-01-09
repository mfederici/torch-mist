from typing import Optional

import torch

from torch_mist.baseline import BatchLogMeanExp
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.implementations.js import JS
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)


class SMILE(JS, BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
        tau: float = 5.0,
    ):
        JS.__init__(
            self,
            critic=critic,
            neg_samples=neg_samples,
        )
        BaselineDiscriminativeMIEstimator.__init__(
            self,
            critic=critic,
            neg_samples=neg_samples,
            baseline=BatchLogMeanExp("all"),
        )
        self.tau = tau

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return JS.batch_loss(self, x, y)

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return BaselineDiscriminativeMIEstimator.log_ratio(self, x, y)

    def _approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.Tensor,
        log_w: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Clamp f_ between [-\tau, \tau] (before re-weighting)
        return super()._approx_log_partition(
            x=x,
            y=y,
            f_=torch.clamp(f_, min=-self.tau, max=self.tau),
            log_w=log_w,
        )
