from typing import Optional, List, Any

import pyro.distributions.transforms as pyro_transforms_module
from pyro.nn import DenseNN

import core.distributions.transforms as transforms_module
from core.distributions.conditional import ConditionalCategorical

from core.distributions.transforms import ConditionalTransformedDistributionModule
from core.distributions.parametrizations import ParametrizedNormal, ConstantParametrizedConditionalDistribution


def conditional_transformed_normal(
        input_dim: int,
        context_dim: int,
        hidden_dims: List[int],
        transform_name: str = "conditional_linear",
        n_transforms: int = 1,
        **transform_params

):

    base = ConstantParametrizedConditionalDistribution(input_dim=input_dim, parametrization=ParametrizedNormal())
    transforms = []
    if hasattr(pyro_transforms_module, transform_name):
        transform_factory = getattr(pyro_transforms_module, transform_name)
    elif hasattr(transforms_module, transform_name):
        transform_factory = getattr(transforms_module, transform_name)
    else:
        raise NotImplementedError(
            f"Transform {transform_name} is not implemented."
        )
    for transform in range(n_transforms):
        transform = transform_factory(input_dim=input_dim, context_dim=context_dim, hidden_dims=hidden_dims, **transform_params)
        transforms.append(transform)

    return ConditionalTransformedDistributionModule(base_dist=base, transforms=transforms)


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
