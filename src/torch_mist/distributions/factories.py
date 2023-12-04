import inspect
from typing import List, Union, Dict

from pyro.distributions import ConditionalTransform, ConditionalDistribution
from pyro.nn import DenseNN

from torch.distributions import Transform, Distribution

from torch_mist.distributions.categorical import ConditionalCategoricalModule
from torch_mist.distributions.joint.base import JointDistribution
from torch_mist.distributions.normal import (
    ConditionalTransformedNormalModule,
    TransformedNormalModule,
    JointTransformedNormalModule,
)

from torch_mist.distributions.transforms.utils import fetch_transform


def make_transforms(
    input_dim: int,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
    **kwargs,
) -> List[Union[Transform, ConditionalTransform]]:
    assert n_transforms > 0, "n_transforms must be greater than 0"
    transforms = []

    transform_factory = fetch_transform(transform_name)

    # Check the arguments
    kwargs_to_delete = []
    for arg_name in kwargs:
        if not (arg_name in inspect.signature(transform_factory).parameters):
            print(
                f"Warning: parameter {arg_name} ignored for {transform_name}."
            )
            kwargs_to_delete.append(arg_name)

    for arg_name in kwargs_to_delete:
        del kwargs[arg_name]

    for transform in range(n_transforms):
        transform = transform_factory(input_dim=input_dim, **kwargs)
        transforms.append(transform)

    return transforms


def transformed_normal(
    input_dim: int,
    transform_name: str = "linear",
    n_transforms: int = 1,
    **kwargs,
) -> Distribution:
    assert n_transforms > 0, "n_transforms must be greater than 0"

    transforms = make_transforms(
        input_dim=input_dim,
        transform_name=transform_name,
        n_transforms=n_transforms,
        **kwargs,
    )

    return TransformedNormalModule(
        input_dim=input_dim,
        transforms=transforms,
    )


def conditional_transformed_normal(
    input_dim: int,
    context_dim: int,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
    **kwargs,
) -> ConditionalDistribution:
    assert n_transforms > 0, "n_transforms must be greater than 0"

    transforms = make_transforms(
        input_dim=input_dim,
        context_dim=context_dim,
        transform_name=transform_name,
        n_transforms=n_transforms,
        **kwargs,
    )

    return ConditionalTransformedNormalModule(
        input_dim=input_dim, transforms=transforms
    )


def joint_transformed_normal(
    input_dims: Dict[str, int],
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
    **kwargs,
) -> JointDistribution:
    assert n_transforms > 0, "n_transforms must be greater than 0"

    input_dim = sum(input_dims.values())

    transforms = make_transforms(
        input_dim=input_dim,
        transform_name=transform_name,
        n_transforms=n_transforms,
        **kwargs,
    )

    return JointTransformedNormalModule(
        input_dims=input_dims, transforms=transforms
    )


def conditional_categorical(
    n_classes: int,
    context_dim: int,
    hidden_dims: List[int],
    temperature: float = 1.0,
):
    net = DenseNN(
        input_dim=context_dim, hidden_dims=hidden_dims, param_dims=[n_classes]
    )
    return ConditionalCategoricalModule(net, temperature=temperature)
