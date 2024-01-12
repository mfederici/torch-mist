from typing import Dict

import torch


from torch_mist.distributions.conditional.unconditional import (
    UnconditionalDistribution,
)
from torch_mist.distributions.empirical import EmpiricalDistribution
from torch_mist.estimators.generative.base import (
    ConditionalGenerativeMIEstimator,
)
from torch_mist.utils.shape import expand_to_same_shape


class DummyGenerativeMIEstimator(ConditionalGenerativeMIEstimator):
    lower_bound: bool = True
    infomax_gradient: Dict[str, bool] = {"x": True, "y": False}

    def __init__(
        self,
    ):
        super().__init__(
            q_Y_given_X=UnconditionalDistribution(EmpiricalDistribution())
        )

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = expand_to_same_shape(x, y)
        return x[..., 0] * 0

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_ratio(x, y)
