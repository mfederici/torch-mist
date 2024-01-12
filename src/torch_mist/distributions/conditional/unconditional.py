import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution


class UnconditionalDistribution(ConditionalDistribution):
    def __init__(self, marginal_distribution: Distribution):
        super().__init__()
        self.marginal_distribution = marginal_distribution

    def condition(self, context: torch.Tensor) -> Distribution:
        return self.marginal_distribution

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += (
            "    (marginal_distribution): "
            + self.marginal_distribution.__repr__().replace("\n", "\n  ")
        )
        s += "  \n)"
        return s
