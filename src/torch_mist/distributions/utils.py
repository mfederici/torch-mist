from typing import List

from pyro.nn import DenseNN

from .conditional import ConditionalCategorical

# from core.distributions.transforms import ConditionalTransformedDistributionModule
# from core.distributions.parametrizations import ParametrizedNormal, ConstantParametrizedConditionalDistribution
from torch.distributions import Distribution, Independent
from torch_mist.distributions.parametrizations.map import LocScaleMap
from torch import nn

from typing import Dict, List, Any, Optional

import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution


from torch_mist.distributions.transforms import ConditionalTransformedDistributionModule, DistributionModule
from torch_mist.distributions.parametrizations.map import LocScaleMap


def fetch_transform(transform_name: str):
    import pyro.distributions.transforms as pyro_transforms_module
    import torch_mist.distributions.transforms as transforms_module

    if hasattr(pyro_transforms_module, transform_name):
        transform_factory = getattr(pyro_transforms_module, transform_name)
    elif hasattr(transforms_module, transform_name):
        transform_factory = getattr(transforms_module, transform_name)
    else:
        raise NotImplementedError(
            f"Transform {transform_name} is not implemented."
        )
    return transform_factory


class ParametricNormal(Distribution, nn.Module):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        nn.Module.__init__(self)
        Distribution.__init__(self, validate_args=False)
        self.loc = nn.Parameter(loc)
        self.log_scale = nn.Parameter(scale.log())
        self.parametrization = LocScaleMap()

    def rsample(self, sample_shape=torch.Size()):
        return Normal(**self.parametrization([self.loc, self.log_scale])).rsample(sample_shape)

    def log_prob(self, value):
        return Independent(Normal(**self.parametrization([self.loc, self.log_scale])),1).log_prob(value)

def conditional_transformed_normal(
    input_dim: int,
    context_dim: int,
    hidden_dims: Optional[List[int]] = None,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
    transform_params: Dict[Any, Any] = None
):
    if transform_params is None:
        transform_params = {}

    base = ParametricNormal(torch.zeros(input_dim), torch.zeros(input_dim))
    transforms = []

    transform_factory = fetch_transform(transform_name)

    for transform in range(n_transforms):
        transform = transform_factory(
            input_dim=input_dim,
            context_dim=context_dim,
            hidden_dims=hidden_dims,
            **transform_params
        )

        transforms.append(transform)

    return ConditionalTransformedDistributionModule(base_dist=base, transforms=transforms)


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


def flow_transformed_normal(
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        transform_name: str = "linear",
        n_transforms: int = 1,
        transform_params: Dict[Any, Any] = None,
) -> FlowTransformedNormal:
    return FlowTransformedNormal(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
        transform_params=transform_params
    )


def conditional_categorical(
    n_classes: int,
    context_dim: int,
    hidden_dims: List[int],
):
    net = DenseNN(input_dim=context_dim, hidden_dims=hidden_dims, param_dims=[n_classes])
    return ConditionalCategorical(net)

def conditional_mutivariate_categorical(
    n_classes: int,
    output_dim: int,
    context_dim: int,
    hidden_dims: List[int],
):
    net = DenseNN(input_dim=context_dim, hidden_dims=hidden_dims, param_dims=[n_classes] * output_dim)
    return ConditionalCategorical(net)
