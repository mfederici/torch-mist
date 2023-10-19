from typing import List, Dict, Any

import torch
import math

from torch_mist.baselines import BatchLogMeanExp
from torch_mist.critic.base import CRITIC_TYPE, JOINT_CRITIC, SEPARABLE_CRITIC
from torch_mist.critic.separable import SeparableCritic
from torch_mist.estimators.discriminative.baseline import (
    BaselineDiscriminativeMIEstimator,
)


class InfoNCE(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: SeparableCritic,
    ):
        # Note that this can be equivalently obtained by extending TUBA with a BatchLogMeanExp(dim=1) baseline
        # This implementation saves some computation
        super().__init__(
            critic=critic,
            neg_samples=0,  # 0 signifies the whole batch is used as negative samples
            baseline=BatchLogMeanExp("first"),
        )

    def compute_log_normalization(
        self, x: torch.Tensor, y: torch.Tensor, f_: torch.tensor
    ):
        # We override the compute_log_normalization just for efficiency
        # The result would be the same as the TUBA implementation with BatchLogMeanExp('first') baseline
        log_norm = f_.logsumexp(0) - math.log(f_.shape[0])

        # We override the compute_log_normalization for efficiency since e^(F(x,y))-b(x) = 1
        log_norm = self.baseline(x=x, y=y, f_=f_)
        return log_norm


def infonce(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    critic_type: str = SEPARABLE_CRITIC,
    **kwargs
) -> InfoNCE:
    from torch_mist.critic.utils import critic_nn

    return InfoNCE(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            critic_type=critic_type,
            hidden_dims=hidden_dims,
            **kwargs
        )
    )
