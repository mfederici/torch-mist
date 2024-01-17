from typing import Optional

import torch

from torch_mist.baseline import BatchLogMeanExp
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)


class InfoNCE(BaselineDiscriminativeMIEstimator):
    def __init__(self, critic: Critic, neg_samples: int = 0):
        # Note that this can be equivalently obtained by extending TUBA with a BatchLogMeanExp(dim=1) baseline
        # This implementation saves some computation
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,  # 0 signifies the whole batch is used as negative samples
            baseline=BatchLogMeanExp("first"),
        )

    def _approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.tensor,
        log_w: Optional[torch.Tensor],
    ):
        # Add the log_weights if provided
        if not (log_w is None):
            assert log_w.ndim == f_.ndim
            f_ = f_ + log_w

        # We override the compute_log_normalization just for efficiency
        # The result would be the same as the TUBA implementation with BatchLogMeanExp('first') baseline
        # We override the compute_log_normalization for efficiency since e^(F(x,y))-b(x) = 1
        log_Z = self.baseline(x=x, f_=f_)
        return log_Z
