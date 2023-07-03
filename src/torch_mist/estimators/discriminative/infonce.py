from typing import List, Dict, Any

import torch
import math
from torch_mist.critic.utils import critic
from torch_mist.critic.separable import SeparableCritic
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator
from torch_mist.utils.caching import cached


class InfoNCE(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            critic: SeparableCritic,
    ):
        # Note that this can be equivalently obtained by extending TUBA with a BatchLogMeanExp(dim=1) baseline
        # This implementation saves some computation
        super().__init__(
            critic=critic,
            mc_samples=0,  # 0 signifies the whole batch is used as negative samples
        )

    @cached
    def compute_log_ratio(self, x: torch.Tensor, y: torch.Tensor, f: torch.Tensor, f_: torch.tensor):
        # f has shape [N, ...]
        # f_ has shape [N, N, ...]
        log_norm = f_.logsumexp(0) - math.log(f_.shape[0])
        return f-log_norm


def infonce(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='separable',
        critic_params: Dict[str, Any] = None,
) -> InfoNCE:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    return InfoNCE(
        critic=url_nn,
    )
