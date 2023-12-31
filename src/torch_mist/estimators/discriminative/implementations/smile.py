from typing import Optional

import torch

from torch_mist.baseline import BatchLogMeanExp
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.implementations.js import JS
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
    CombinedDiscriminativeMIEstimator,
)


class SMILE(CombinedDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
        tau: float = 5.0,
    ):
        super().__init__(
            train_estimator=JS(critic=critic, neg_samples=neg_samples),
            eval_estimator=BaselineDiscriminativeMIEstimator(
                critic=critic,
                neg_samples=neg_samples,
                baseline=BatchLogMeanExp("all"),
            ),
        )
        self.tau = tau

    def _approx_log_partition(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        return super()._approx_log_partition(
            x=x, f_=torch.clamp(f_, min=-self.tau, max=self.tau)
        )
