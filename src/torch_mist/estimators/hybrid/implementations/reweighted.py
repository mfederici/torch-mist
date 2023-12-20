from functools import lru_cache
from typing import Optional, Tuple

import torch

from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid.base import HybridMIEstimator


class ReweighedHybridMIEstimator(HybridMIEstimator):
    def sample_negatives(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y_, _ = DiscriminativeMIEstimator.sample_negatives(x, y)
        # Re-weight the terms according to the weighs exp(log p(y|x)/p(y))
        w = self.generative_estimator.log_ratio(x.unsqueeze(0), y_).exp()
        return y_, w
