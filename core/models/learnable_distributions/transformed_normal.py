from typing import Dict, List, Any, Optional

import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution
import pyro.distributions.transforms as pyro_transforms_module

import core.distributions.transforms as transforms_module
from core.distributions.joint.base import JointDistribution

from core.distributions.transforms import ConditionalTransformedDistributionModule, DistributionModule
from core.distributions.parametrizations import ParametrizedNormal, ConstantParametrizedConditionalDistribution


def fetch_transform(transform_name: str):
    if hasattr(pyro_transforms_module, transform_name):
        transform_factory = getattr(pyro_transforms_module, transform_name)
    elif hasattr(transforms_module, transform_name):
        transform_factory = getattr(transforms_module, transform_name)
    else:
        raise NotImplementedError(
            f"Transform {transform_name} is not implemented."
        )
    return transform_factory


class ConditionalFlowTransformedNormal(ConditionalTransformedDistributionModule):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: Optional[List[int]] = None,
        transform_name: str = "conditional_linear",
        n_transforms: int = 1,
        transform_params: Dict[Any, Any] = None
    ):
        if transform_params is None:
            transform_params = {}

        base = ConstantParametrizedConditionalDistribution(input_dim=y_dim, parametrization=ParametrizedNormal())
        transforms = []

        transform_factory = fetch_transform(transform_name)

        for transform in range(n_transforms):
            transform = transform_factory(
                input_dim=y_dim,
                context_dim=x_dim,
                hidden_dims=hidden_dims,
                **transform_params
            )

            transforms.append(transform)

        super().__init__(base_dist=base, transforms=transforms)


class FlowTransformedNormal(DistributionModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        transform_name: str = "linear",
        n_transforms: int = 1,
        transform_params: Dict[Any, Any] = None
    ):
        super().__init__(validate_args=False)

        if transform_params is None:
            transform_params = {}

        self.register_buffer('loc', torch.zeros(input_dim))

        self.transforms = nn.ModuleList()

        transform_factory = fetch_transform(transform_name)

        for transform in range(n_transforms):
            transform = transform_factory(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                **transform_params
            )

            self.transforms.append(transform)

    @property
    def dist(self):
        return TransformedDistribution(Normal(self.loc, 1), [t for t in self.transforms])

    def rsample(self, sample_shape=torch.Size()):
        return self.dist.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)

    def log_prob(self, value):
        return self.dist.log_prob(value)


class FlowTransformedJointNormal(JointDistribution, nn.Module):
    labels = ['x', 'y']
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: Optional[List[int]] = None,
        transform_name: str = "linear",
        n_transforms: int = 1,
        transform_params: Dict[Any, Any] = None
    ):
        nn.Module.__init__(self)
        JointDistribution.__init__(self)

        if transform_params is None:
            transform_params = {}

        self.register_buffer('loc', torch.zeros(x_dim+y_dim))

        self.transforms = nn.ModuleList()

        transform_factory = fetch_transform(transform_name)

        for transform in range(n_transforms):
            transform = transform_factory(
                input_dim=x_dim+y_dim,
                hidden_dims=hidden_dims,
                **transform_params
            )

            self.transforms.append(transform)
        self.x_dim = x_dim
        self.y_dim = y_dim

    @property
    def dist(self):
        return TransformedDistribution(Normal(self.loc, 1), [t for t in self.transforms])

    def rsample(self, sample_shape=torch.Size()):
        sample = self.dist.rsample(sample_shape)
        x, y = torch.split(sample, [self.x_dim, self.y_dim], dim=-1)
        return {'x': x, 'y': y}

    def sample(self, sample_shape=torch.Size()):
        sample = self.dist.sample(sample_shape)
        x, y = torch.split(sample, [self.x_dim, self.y_dim], dim=-1)
        return {'x': x, 'y': y}

    def log_prob(self, value):
        all_values = torch.cat([value['x'], value['y']], dim=-1)
        return self.dist.log_prob(all_values)