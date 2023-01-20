from typing import Dict, List, Any

import pyro.distributions.transforms as pyro_transforms_module
from pyro.nn import DenseNN

import core.distributions.transforms as transforms_module
from core.distributions.conditional import ConditionalCategorical

# from core.distributions.transforms import ConditionalTransformedDistributionModule
# from core.distributions.parametrizations import ParametrizedNormal, ConstantParametrizedConditionalDistribution


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
