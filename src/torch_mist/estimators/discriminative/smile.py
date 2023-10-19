from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import BatchLogMeanExp
from torch_mist.critic.base import Critic, CRITIC_TYPE, JOINT_CRITIC
from torch_mist.estimators.discriminative.js import JS
from torch_mist.estimators.discriminative.baseline import (
    BaselineDiscriminativeMIEstimator,
)


class SMILE(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
        tau: float = 5.0,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
            baseline=BatchLogMeanExp("all"),
        )
        self.tau = tau

    def compute_log_normalization(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        return super().compute_log_normalization(
            x=x, y=y, f_=torch.clamp(f_, min=-self.tau, max=self.tau)
        )

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return JS.loss(self, x, y)


def smile(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    tau: float = 5.0,
    critic_type: str = JOINT_CRITIC,
    **kwargs
) -> SMILE:
    from torch_mist.critic.utils import critic_nn

    return SMILE(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs
        ),
        neg_samples=neg_samples,
        tau=tau,
    )
