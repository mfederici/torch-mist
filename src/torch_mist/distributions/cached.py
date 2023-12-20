from functools import lru_cache

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
