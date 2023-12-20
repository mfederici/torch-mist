from functools import lru_cache
from typing import Tuple, Optional

import torch

from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid.base import HybridMIEstimator
from torch_mist.estimators.transformed.implementations.pq import PQ


class PQHybridMIEstimator(HybridMIEstimator):
    def __init__(
        self,
        pq_estimator: PQ,
        discriminative_estimator: DiscriminativeMIEstimator,
    ):
        super().__init__(
            generative_estimator=pq_estimator,
            discriminative_estimator=discriminative_estimator,
        )

    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        Q_y = self.generative_estimator.transforms["y"](y)
        # Check the labels are the same
        Q_y0 = Q_y[0]
        assert torch.sum(Q_y0.unsqueeze(0) == Q_y) == Q_y.numel()
        return DiscriminativeMIEstimator.sample_negatives(self, x, y)
