from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from torch_mist.baselines import ConstantBaseline
from torch_mist.critic.base import Critic, CRITIC_TYPE, JOINT_CRITIC
from torch_mist.estimators.discriminative.baseline import (
    BaselineDiscriminativeMIEstimator,
)
from torch_mist.utils.caching import reset_cache_before_call


class JS(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
            baseline=ConstantBaseline(value=0.0),
        )

    @reset_cache_before_call
    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Compute the critic on the positives. It has shape [...]
        f = self.critic_on_positives(x=x, y=y)
        # Compute the critic on the negatives. It has shape [M, ...] with M as the number of negative samples
        f_ = self.critic_on_negatives(x=x, y=y)

        loss = F.softplus(-f).mean() + F.softplus(f_).mean()
        return loss


def js(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    critic_type: str = JOINT_CRITIC,
    **kwargs
) -> JS:
    from torch_mist.critic.utils import critic_nn

    return JS(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs
        ),
        neg_samples=neg_samples,
    )
