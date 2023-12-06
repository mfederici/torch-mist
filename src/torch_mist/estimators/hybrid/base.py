from typing import Optional

import torch

from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.estimators.generative.base import GenerativeMIEstimator
from torch_mist.utils.freeze import is_frozen


class HybridMIEstimator(MIEstimator):
    def __init__(
        self,
        generative_estimator: Optional[GenerativeMIEstimator] = None,
        discriminative_estimator: Optional[DiscriminativeMIEstimator] = None,
    ):
        super().__init__()
        self.generative_estimator = generative_estimator
        self.discriminative_estimator = discriminative_estimator

    def unnormalized_log_ratio(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        log_ratio = self.generative_estimator.log_ratio(x, y)
        log_ratio += self.discriminative_estimator.unnormalized_log_ratio(x, y)
        return log_ratio

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_ratio = self.generative_estimator.log_ratio(x, y)
        proposal = self.generative_estimator.q_Y_given_x(x=x)
        self.discriminative_estimator.proposal = proposal
        log_ratio += self.discriminative_estimator.log_ratio(x, y)
        return log_ratio

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if is_frozen(self.generative_estimator):
            e1 = 0.0
        else:
            e1 = self.generative_estimator.batch_loss(x, y)
        if is_frozen(self.discriminative_estimator):
            e2 = 0.0
        else:
            proposal = self.generative_estimator.q_Y_given_x(x=x)
            self.discriminative_estimator.proposal = proposal
            e2 = self.discriminative_estimator.batch_loss(x, y)
        return e1 + e2
