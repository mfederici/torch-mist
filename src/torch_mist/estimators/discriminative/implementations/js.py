from typing import Optional

import torch
import torch.nn.functional as F

from torch_mist.baseline import ConstantBaseline
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
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
    def batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Compute the critic on the positives. It has shape [...]
        f = self.unnormalized_log_ratio(x=x, y=y)
        # Compute the critic on the negatives. It has shape [M, ...] with M as the number of negative samples
        f_ = self.critic_on_negatives(x=x, y=y)

        loss = F.softplus(-f) + F.softplus(f_).mean(0)
        return loss
