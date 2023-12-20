from typing import Optional

import torch
import torch.nn.functional as F

from torch_mist.baseline import ConstantBaseline
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)


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

    def batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Compute the critic on the positives. It has shape [...]
        f = self.unnormalized_log_ratio(x=x, y=y)

        y_, w = self.sample_negatives(x, y)
        f_ = self.critic(x, y_)

        pos = F.softplus(-f)
        neg = F.softplus(f_)

        # Compute the expectation w.r.t the M negatives
        if not (w is None):
            neg = w * neg

        neg = neg.mean(0)

        loss = pos + neg
        return loss
