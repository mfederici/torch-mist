import torch
from pyro.distributions import Delta
from torch.distributions import Distribution, Independent

from core.distributions.parametrizations.base import ParametrizedDistribution


class ParametrizedDelta(ParametrizedDistribution):
    def __init__(self):
        super(ParametrizedDelta, self).__init__(n_params=1)

    def forward(self, x: torch.Tensor) -> Distribution:
        dist = Delta(v=x)
        return Independent(dist, 1)
