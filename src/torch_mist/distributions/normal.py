from typing import List, Union, Dict

import torch

from torch import nn
from torch.distributions import Transform, Distribution, Normal, Independent
from pyro.distributions import ConditionalTransform

from torch_mist.distributions.joint.wrapper import TorchJointDistribution
from torch_mist.distributions.parametrizations.map import LocScaleMap
from torch_mist.distributions.transforms import (
    TransformedDistributionModule,
    ConditionalTransformedDistributionModule,
    ConditionalDistributionModule,
)


class NormalModule(Distribution, nn.Module):
    def __init__(
        self, loc: torch.Tensor, scale: torch.Tensor, learnable: bool = False
    ):
        assert loc.ndim == 1
        nn.Module.__init__(self)
        Distribution.__init__(
            self, event_shape=torch.Size([loc.shape[0]]), validate_args=False
        )
        if learnable:
            self.loc = nn.Parameter(loc)
            self.log_scale = nn.Parameter(scale.log())
        else:
            self.register_buffer("loc", loc)
            self.register_buffer("log_scale", scale.log())
        self.parametrization = LocScaleMap()

    def rsample(self, sample_shape=torch.Size()):
        params = self.parametrization([self.loc, self.log_scale])
        return Normal(**params).rsample(sample_shape)

    def log_prob(self, value):
        params = self.parametrization([self.loc, self.log_scale])
        return Independent(Normal(**params), 1).log_prob(value)

    def __repr__(self):
        return "Normal()"


class StandardNormalModule(NormalModule):
    def __init__(self, n_dim: int):
        super().__init__(loc=torch.zeros(n_dim), scale=torch.ones(n_dim))


class ConditionalStandardNormalModule(ConditionalDistributionModule):
    def __init__(self, n_dim: int):
        super().__init__()
        self.register_buffer("loc", torch.zeros(n_dim))
        self.register_buffer("log_scale", torch.zeros(n_dim))
        self.parametrization = LocScaleMap()

    def condition(self, context):
        extra_dims = context.ndim - self.loc.ndim
        loc, log_scale = self.loc, self.log_scale
        for _ in range(extra_dims):
            loc = loc.unsqueeze(0)
            log_scale = log_scale.unsqueeze(0)

        loc = loc.expand(*context.shape[:-1], -1)
        log_scale = log_scale.expand(*context.shape[:-1], -1)

        return Independent(Normal(**self.parametrization([loc, log_scale])), 1)


class TransformedNormalModule(TransformedDistributionModule):
    def __init__(self, input_dim: int, transforms: List[Transform]):
        super().__init__(
            base_dist=StandardNormalModule(n_dim=input_dim),
            transforms=transforms,
        )


class ConditionalTransformedNormalModule(
    ConditionalTransformedDistributionModule
):
    def __init__(
        self,
        input_dim: int,
        transforms: List[Union[Transform, ConditionalTransform]],
    ):
        super().__init__(
            base_dist=ConditionalStandardNormalModule(n_dim=input_dim),
            transforms=transforms,
        )


class JointTransformedNormalModule(TorchJointDistribution):
    def __init__(
        self,
        input_dims: Dict[str, int],
        transforms: List[Union[Transform, ConditionalTransform]],
    ):
        splits: List[int] = []
        variables: List[str] = []
        for variable, dim in input_dims.items():
            splits.append(dim)
            variables.append(variable)

        super().__init__(
            torch_dist=TransformedNormalModule(
                input_dim=sum(splits),
                transforms=transforms,
            ),
            variables=variables,
            splits=splits,
            split_dim=-1,
        )
