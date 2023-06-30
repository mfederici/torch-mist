from typing import List, Dict, Any, Optional

import torch

from torch_mist.critic import Critic, critic
from torch_mist.estimators.discriminative.js import JS
from torch_mist.estimators.discriminative.mine import MINE


class SMILE(MINE, JS):
    def __init__(
            self,
            critic: Critic,
            mc_samples: int = 1,
            tau: float = 5.0,
    ):
        MINE.__init__(
            self,
            critic=critic,
            mc_samples=mc_samples,
        )
        self.tau = tau

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        return MINE._compute_log_ratio(
            self,
            x=x, y=y, f=f, y_=y_,
            f_=torch.clamp(f_, -self.tau, self.tau)
        )

    def _compute_log_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return JS._compute_log_ratio_grad(self, x=x, y=y, f=f, y_=y_, f_=f_)


def smile(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        mc_samples: int = 1,
        tau: float = 5.0,
        critic_params: Dict[str, Any] = None,
) -> SMILE:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    return SMILE(
        critic=url_nn,
        mc_samples=mc_samples,
        tau=tau,
    )
