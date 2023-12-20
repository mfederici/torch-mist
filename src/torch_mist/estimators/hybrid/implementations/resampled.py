from functools import lru_cache
from typing import Optional, Tuple

import torch

from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.estimators.generative import ConditionalGenerativeMIEstimator
from torch_mist.estimators.hybrid.base import HybridMIEstimator


class ResampledHybridMIEstimator(HybridMIEstimator):
    def __init__(
        self,
        generative_estimator: ConditionalGenerativeMIEstimator,
        discriminative_estimator: DiscriminativeMIEstimator,
    ):
        assert isinstance(
            generative_estimator, ConditionalGenerativeMIEstimator
        )
        super().__init__(generative_estimator, discriminative_estimator)

    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        N = x.shape[0]
        neg_samples = self.n_negatives_to_use(N)

        assert isinstance(
            self.generative_estimator, ConditionalGenerativeMIEstimator
        )
        q_Y_given_x = self.generative_estimator.q_Y_given_X.condition(x)
        # Sample from the proposal r(y|x) [M, ..., Y_DIM] with M as the number of neg_samples
        y_ = q_Y_given_x.sample(sample_shape=torch.Size([neg_samples]))
        # The shape of the samples from the proposal distribution is [M, ..., Y_DIM]
        assert y_.ndim == x.ndim + 1 and y_.shape[0] == neg_samples
        assert y_.shape[0] == neg_samples and y_.ndim == x.ndim + 1

        return y_, None
