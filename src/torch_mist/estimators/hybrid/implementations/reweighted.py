from functools import lru_cache
from typing import Optional, Tuple

import torch

from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid.base import HybridMIEstimator
from torch_mist.utils.caching import cached_method


class ReweighedHybridMIEstimator(HybridMIEstimator):
    @cached_method
    def sample_negatives(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        x_, y_, old_log_w = DiscriminativeMIEstimator.sample_negatives(
            self, x, y
        )
        assert old_log_w is None

        # Re-weight the terms according to the weights log q(y|x)/p(y)
        log_w = self.generative_estimator.log_ratio(x_, y_).detach()
        return x_, y_, log_w
