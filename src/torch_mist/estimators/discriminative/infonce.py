from typing import List, Dict, Any

import torch
import math
from torch_mist.critic import critic, SeparableCritic
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator



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

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        N, M = f_.shape[0], f_.shape[1]

        # Compute the estimation for the normalization constant
        # log 1/M \sum_{j=1}^M f_[i,j] = logsumexp(f_,1).mean(0) - log M
        log_Z_value = (torch.logsumexp(f_, 1) - math.log(M)).unsqueeze(1)

        log_ratio = f - log_Z_value

        return log_ratio


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
