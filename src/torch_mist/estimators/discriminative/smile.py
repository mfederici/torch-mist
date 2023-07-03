from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import BatchLogMeanExp
from torch_mist.critic.base import Critic
from torch_mist.critic.utils import critic
from torch_mist.estimators.discriminative.js import JS
from torch_mist.estimators.discriminative.tuba import TUBA


class SMILE(TUBA):
    def __init__(
            self,
            critic: Critic,
            mc_samples: int = 1,
            tau: float = 5.0,
    ):
        super().__init__(
            critic=critic,
            mc_samples=mc_samples,
            baseline=BatchLogMeanExp('all'),
        )
        self.tau = tau

    def compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor,
    ) -> torch.Tensor:
        return super().compute_log_ratio(
            x=x, y=y, f=f,
            f_=torch.clamp(f_, min=-self.tau, max=self.tau)
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
        critic_type: str = 'joint',
        mc_samples: int = 1,
        tau: float = 5.0,
        critic_params: Dict[str, Any] = None,
) -> SMILE:
    critic_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    return SMILE(
        critic=critic_nn,
        mc_samples=mc_samples,
        tau=tau,
    )
