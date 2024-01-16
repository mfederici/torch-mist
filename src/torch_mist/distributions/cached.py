from typing import Any

import torch
from torch import nn
from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from torch_mist.distributions.transforms import ConditionalDistributionModule
from torch_mist.utils.caching import cached


class CachedConditionalDistribution(ConditionalDistributionModule):
    def __init__(self, conditional_dist: ConditionalDistribution):
        super().__init__()
        self.conditional_dist = conditional_dist

    @cached
    def condition(self, condition: Any) -> Distribution:
        return self.conditional_dist.condition(condition)

    def __repr__(self):
        return "Cached_" + self.conditional_dist.__repr__()


class CachedDistribution(Distribution, nn.Module):
    def __init__(self, distribution: Distribution):
        nn.Module.__init__(self)
        Distribution.__init__(
            self,
            validate_args=False,
            batch_shape=distribution.batch_shape,
            event_shape=distribution.event_shape,
        )
        self.distribution = distribution

    @cached
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(value)

    def __repr__(self):
        return "Cached_" + self.distribution.__repr__()
