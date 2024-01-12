from functools import lru_cache

import torch
from torch import nn
from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from torch_mist.distributions.transforms import ConditionalDistributionModule


class CachedConditionalDistribution(ConditionalDistributionModule):
    def __init__(self, conditional_dist: ConditionalDistribution):
        super().__init__()
        self.conditional_dist = conditional_dist

    @lru_cache(maxsize=1)
    def condition(self, *args, **kwargs) -> Distribution:
        return self.conditional_dist.condition(*args, **kwargs)

    def __repr__(self):
        return self.conditional_dist.__repr__()


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

    @lru_cache(maxsize=1)
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(value)

    def __repr__(self):
        return "Cached_" + self.distribution.__repr__()
