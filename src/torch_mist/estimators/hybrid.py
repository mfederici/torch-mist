from typing import Optional

import torch

from torch_mist.estimators.base import MutualInformationEstimator
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator
from torch_mist.estimators.generative.base import GenerativeMutualInformationEstimator
from torch_mist.utils.caching import reset_cache_before_call


class HybridMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            generative_estimator: Optional[GenerativeMutualInformationEstimator] = None,
            discriminative_estimator: Optional[DiscriminativeMutualInformationEstimator] = None,
            train_generative_estimator: bool = True,
            train_discriminative_estimator: bool = True,
    ):
        super().__init__()
        self.generative_estimator = generative_estimator
        self.discriminative_estimator = discriminative_estimator
        self.train_generative_estimator = train_generative_estimator
        self.train_discriminative_estimator = train_discriminative_estimator

    def expected_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:

        e1 = self.generative_estimator.expected_log_ratio(x, y)
        proposal = self.generative_estimator.q_Y_given_x(x=x)
        self.discriminative_estimator.proposal = proposal
        e2 = self.discriminative_estimator.expected_log_ratio(x, y)

        return e1 + e2

    def unnormalized_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        log_ratio = self.generative_estimator.log_ratio(x, y)
        proposal = self.generative_estimator.q_Y_given_x(x=x)
        self.discriminative_estimator.proposal = proposal
        log_ratio += self.discriminative_estimator.unnormalized_log_ratio(x, y)

        return log_ratio

    @reset_cache_before_call
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.train_generative_estimator:
            e1 = self.generative_estimator.loss(x, y)
        else:
            e1 = 0.0
        proposal = self.generative_estimator.q_Y_given_x(x=x)
        self.discriminative_estimator.proposal = proposal
        if self.train_discriminative_estimator:
            e2 = self.discriminative_estimator.loss(x, y)
        else:
            e2 = 0.0
        return e1 + e2
