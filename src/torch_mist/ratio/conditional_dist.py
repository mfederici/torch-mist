import torch

from pyro.distributions import ConditionalDistribution

from .base import UnnormalizedRatioEstimator


class ConditionalDistributionRatio(UnnormalizedRatioEstimator):
    def __init__(self, conditional_y_x: ConditionalDistribution):
        super().__init__()
        self.conditional_y_x = conditional_y_x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.conditional_y_x.condition(x.unsqueeze(1)).log_prob(y)
