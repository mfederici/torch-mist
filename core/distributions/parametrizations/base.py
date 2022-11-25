from typing import Union, List

import torch
import torch.nn as nn

from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution


class ParametrizedDistribution(nn.Module):
    def __init__(self, n_params):
        super(ParametrizedDistribution, self).__init__()
        self.n_params = n_params

    def forward(self, x: torch.Tensor) -> Distribution:
        raise NotImplementedError()


class ConstantParametrizedConditionalDistribution(ConditionalDistribution, nn.Module):
    def __init__(self,
                 input_dim: int,
                 parametrization: ParametrizedDistribution,
                 initial_value: Union[float, List[float], None] = None,
                 optimize_params: bool = False
                 ):
        super(ConstantParametrizedConditionalDistribution, self).__init__()

        self.parametrization = parametrization

        n_params = input_dim * self.parametrization.n_params

        if initial_value is None:
            params = torch.zeros(n_params)
        elif isinstance(initial_value, list):
            assert len(initial_value) == n_params
            params = torch.Tensor(initial_value)
        else:
            params = torch.zeros(n_params) + initial_value

        if not optimize_params:
            self.register_buffer('params', params)
        else:
            self.params = nn.Parameter(params)

    def condition(self, context):
        unsqueeze_dims = context.ndim - self.params.ndim
        params = self.params
        for i in range(unsqueeze_dims):
            params = params.unsqueeze(0)

        shape = [context.shape[i] if i < unsqueeze_dims else params.shape[i] for i in range(context.ndim)]
        params = params.expand(shape)

        assert params.ndim == context.ndim

        return self.parametrization(params)
