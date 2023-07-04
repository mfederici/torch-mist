from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch_mist.critic.base import Critic
from torch_mist.critic.utils import critic
from torch_mist.estimators.discriminative.nwj import NWJ
from torch_mist.utils.caching import reset_cache_after_call, reset_cache_before_call


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

    @reset_cache_before_call
    def loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        f = self.critic_on_positives(x=x, y=y)
        f_ = self.critic_on_negatives(x=x, y=y)
        loss = F.softplus(-f).mean() + F.softplus(f_).mean()
        return loss


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
