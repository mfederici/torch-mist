from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch_mist.critic import Critic, critic
from torch_mist.estimators.discriminative.nwj import NWJ


class JS(NWJ):
    def __init__(
            self,
            critic: Critic,
            mc_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            mc_samples=mc_samples,
        )

    def _compute_log_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> Optional[torch.Tensor]:
        ratio_grad = F.softplus(-f).mean(1) + F.softplus(f_).mean(1)
        return ratio_grad


def js(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        mc_samples: int = 1,
        critic_params: Dict[str, Any] = None,
) -> JS:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    return JS(
        critic=url_nn,
        mc_samples=mc_samples,
    )
