from typing import Optional

import torch

from src.torch_mist.models.mi_estimator.base import MutualInformationEstimator
from src.torch_mist.models.mi_estimator.discriminative import DiscriminativeMutualInformationEstimator
from src.torch_mist.models.mi_estimator.generative import VariationalProposalMutualInformationEstimator


class HybridMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            generative_estimator: Optional[VariationalProposalMutualInformationEstimator] = None,
            discriminative_estimator: Optional[DiscriminativeMutualInformationEstimator] = None,
    ):
        super().__init__()
        self.generative_estimator = generative_estimator
        self.discriminative_estimator = discriminative_estimator
        self.discriminative_estimator.proposal = self.generative_estimator.conditional_y_x

    def log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ):

        e1 = self.generative_estimator.log_ratio(x, y)
        e2 = self.discriminative_estimator.log_ratio(x, y)

        return e1 + e2
