from typing import Optional

import torch

from torch_mist.critic.constant import ConstantCritic
from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator


class DummyDiscriminativeMIEstimator(DiscriminativeMIEstimator):
    def __init__(
        self,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=ConstantCritic(),
            neg_samples=neg_samples,
        )

    def _approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.Tensor,
        log_w: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return f_[0] * 0
