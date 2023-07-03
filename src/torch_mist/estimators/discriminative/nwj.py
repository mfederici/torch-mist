from typing import List, Dict, Any

import torch
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator
from torch_mist.critic.base import Critic
from torch_mist.critic.utils import critic


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

    def compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        # f has shape [N, ...]
        # f_ has shape [M', N, ...] with M' as the number of mc_samples.
        log_norm = f_.exp().mean(0) - 1
        assert log_norm.shape == f.shape

        return f-log_norm

def nwj(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        mc_samples: int = 1,
        critic_type: str = 'joint',
        critic_params: Dict[str, Any] = None,
) -> NWJ:

    critic_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    return NWJ(
        critic=critic_nn,
        mc_samples=mc_samples,
    )
