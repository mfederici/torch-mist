import inspect
from typing import List, Union, Dict, Optional, Callable, Any

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


def delete_unused_kwargs(
    transform_factory: Callable[[Any], Any],
    all_kwargs: Dict[str, Any],
    warnings: bool = True,
):
    # Check the arguments
    kwargs = {}
    for arg_name, value in all_kwargs.items():
        if arg_name in inspect.signature(transform_factory).parameters:
            kwargs[arg_name] = value
        elif warnings:
            name = transform_factory.__name__
            print(f"Warning: parameter {arg_name} ignored for {name}.")

    return kwargs


def make_transforms(
    input_dim: int,
    transform_name: str = "conditional_linear",
    normalization: Optional[str] = None,
    n_transforms: int = 1,
    **kwargs,
) -> List[Union[Transform, ConditionalTransform]]:
    assert n_transforms > 0, "n_transforms must be greater than 0"
    transforms = []

    kwargs["input_dim"] = input_dim

    transform_factory = fetch_transform(transform_name)
    transform_kwargs = delete_unused_kwargs(
        transform_factory, kwargs, warnings=True
    )

    if normalization:
        norm_factory = fetch_transform(normalization)
        norm_kwargs = delete_unused_kwargs(
            norm_factory, kwargs, warnings=False
        )
    else:
        norm_factory = None
        norm_kwargs = {}

    for transform in range(n_transforms):
        transform = transform_factory(**transform_kwargs)
        transforms.append(transform)
        if norm_factory:
            norm = norm_factory(**norm_kwargs)
            transforms.append(norm)

    return transforms


def transformed_normal(
    input_dim: int,
    transform_name: str = "linear",
    n_transforms: int = 1,
    normalization: Optional[str] = None,
    **kwargs,
) -> Distribution:
    assert n_transforms > 0, "n_transforms must be greater than 0"

    transforms = make_transforms(
        input_dim=input_dim,
        transform_name=transform_name,
        n_transforms=n_transforms,
        normalization=normalization,
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
    normalization: Optional[str] = None,
    **kwargs,
) -> ConditionalDistribution:
    assert n_transforms > 0, "n_transforms must be greater than 0"

    transforms = make_transforms(
        input_dim=input_dim,
        context_dim=context_dim,
        transform_name=transform_name,
        n_transforms=n_transforms,
        normalization=normalization,
        **kwargs,
    )

    return ConditionalTransformedNormalModule(
        input_dim=input_dim, transforms=transforms
    )


def joint_transformed_normal(
    input_dims: Dict[str, int],
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
    normalization: Optional[str] = None,
    **kwargs,
) -> JointDistribution:
    assert n_transforms > 0, "n_transforms must be greater than 0"

    input_dim = sum(input_dims.values())

    transforms = make_transforms(
        input_dim=input_dim,
        transform_name=transform_name,
        n_transforms=n_transforms,
        normalization=normalization,
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
