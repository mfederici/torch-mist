from typing import List, Dict, Any

import torch
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator
from torch_mist.critic import Critic, critic


class NWJ(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            critic: Critic,
            mc_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            mc_samples=mc_samples,
        )

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        log_ratio = f - f_.exp().mean(1, keepdim=True) + 1

        return log_ratio


def nwj(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        mc_samples: int = 1,
        critic_type: str = 'joint',
        critic_params: Dict[str, Any] = None,
) -> NWJ:

    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params

    )

    return NWJ(
        critic=url_nn,
        mc_samples=mc_samples,
    )
