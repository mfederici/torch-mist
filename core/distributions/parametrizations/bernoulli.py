import inspect

import torch
from pyro.distributions import Independent
from torch.distributions import Bernoulli, Distribution

from core.distributions.parametrizations.base import ParametrizedDistribution


class ParametrizedBernoulli(ParametrizedDistribution):
    def __init__(self, param_name='logits'):
        super(ParametrizedDistribution, self).__init__(n_params=1)
        self.param_name = param_name
        assert param_name in inspect.signature(Bernoulli).parameters

    def forward(self, x: torch.Tensor) -> Distribution:
        dist = Bernoulli(**{self.param_name: x})
        return Independent(dist, 1)
