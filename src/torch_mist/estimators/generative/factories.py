from typing import Optional, Union, List

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch_mist.distributions.factories import joint_transformed_normal
from torch_mist.distributions.joint.base import JointDistribution
from torch_mist.estimators.generative.implementations import (
    BA,
    CLUB,
    DoE,
    GM,
    L1Out,
    DummyGenerativeMIEstimator,
)


def ba(
    entropy_y: Union[float, torch.Tensor],
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    q_Y_given_X: Optional[ConditionalDistribution] = None,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
) -> BA:
    from torch_mist.distributions.factories import (
        conditional_transformed_normal,
    )

    if q_Y_given_X is None:
        if x_dim is None or y_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_Y_given_X or x_dim, y_dim and hidden_dims must be provided"
            )

        q_Y_given_X = conditional_transformed_normal(
            input_dim=y_dim,
            context_dim=x_dim,
            hidden_dims=hidden_dims,
            transform_name=transform_name,
            n_transforms=n_transforms,
        )

    if isinstance(entropy_y, float):
        entropy_y = torch.FloatTensor([entropy_y])

    entropy_y = entropy_y.squeeze()

    return BA(
        q_Y_given_X=q_Y_given_X,
        entropy_y=entropy_y,
    )


def club(
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    q_Y_given_X: Optional[ConditionalDistribution] = None,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
) -> CLUB:
    from torch_mist.distributions.factories import (
        conditional_transformed_normal,
    )

    if q_Y_given_X is None:
        if x_dim is None or y_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_Y_given_X or x_dim, y_dim and hidden_dims must be provided"
            )

        q_Y_given_X = conditional_transformed_normal(
            input_dim=y_dim,
            context_dim=x_dim,
            hidden_dims=hidden_dims,
            transform_name=transform_name,
            n_transforms=n_transforms,
        )

    return CLUB(
        q_Y_given_X=q_Y_given_X,
    )


def doe(
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    q_Y_given_X: Optional[ConditionalDistribution] = None,
    q_Y: Optional[Distribution] = None,
    conditional_transform_name: str = "conditional_linear",
    n_conditional_transforms: int = 1,
    marginal_transform_name: str = "linear",
    n_marginal_transforms: int = 1,
) -> DoE:
    from torch_mist.distributions.factories import (
        conditional_transformed_normal,
        transformed_normal,
    )

    if q_Y_given_X is None:
        if x_dim is None or y_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_Y_given_X or x_dim, y_dim and hidden_dims must be specified."
            )
        q_Y_given_X = conditional_transformed_normal(
            input_dim=y_dim,
            context_dim=x_dim,
            hidden_dims=hidden_dims,
            transform_name=conditional_transform_name,
            n_transforms=n_conditional_transforms,
        )

    if q_Y is None:
        if y_dim is None:
            raise ValueError(
                "Either q_Y or y_dim and hidden_dims must be specified."
            )
        q_Y = transformed_normal(
            input_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name=marginal_transform_name,
            n_transforms=n_marginal_transforms,
        )

    return DoE(
        q_Y_given_X=q_Y_given_X,
        q_Y=q_Y,
    )


def dummy_generative() -> DummyGenerativeMIEstimator:
    return DummyGenerativeMIEstimator()


def gm(
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: List[int] = None,
    q_XY: Optional[JointDistribution] = None,
    q_Y: Optional[Distribution] = None,
    q_X: Optional[Distribution] = None,
    joint_transform_name: str = "affine_autoregressive",
    n_joint_transforms: int = 1,
    marginal_transform_name: str = "linear",
    n_marginal_transforms: int = 1,
) -> GM:
    from torch_mist.distributions.factories import transformed_normal

    if q_XY is None:
        if x_dim is None or y_dim is None or hidden_dims is None:
            raise ValueError(
                "x_dim, y_dim, and hidden_dims must be provided if q_XY is not provided"
            )
        q_XY = joint_transformed_normal(
            input_dims={"x": x_dim, "y": y_dim},
            transform_name=joint_transform_name,
            n_transforms=n_joint_transforms,
        )

    if q_X is None:
        if x_dim is None or hidden_dims is None:
            raise ValueError(
                "x_dim and hidden_dims must be provided if q_X is not provided"
            )
        q_X = transformed_normal(
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            transform_name=marginal_transform_name,
            n_transforms=n_marginal_transforms,
        )

    if q_Y is None:
        if y_dim is None or hidden_dims is None:
            raise ValueError(
                "y_dim and hidden_dims must be provided if q_Y is not provided"
            )
        q_Y = transformed_normal(
            input_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name=marginal_transform_name,
            n_transforms=n_marginal_transforms,
        )

    return GM(
        q_XY=q_XY,
        q_X=q_X,
        q_Y=q_Y,
    )


def l1out(
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    q_Y_given_X: Optional[ConditionalDistribution] = None,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
) -> L1Out:
    from torch_mist.distributions.factories import (
        conditional_transformed_normal,
    )

    if q_Y_given_X is None:
        if x_dim is None or y_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_Y_given_X or x_dim, y_dim and hidden_dims must be provided"
            )

        q_Y_given_X = conditional_transformed_normal(
            input_dim=y_dim,
            context_dim=x_dim,
            hidden_dims=hidden_dims,
            transform_name=transform_name,
            n_transforms=n_transforms,
        )

    return L1Out(
        q_Y_given_X=q_Y_given_X,
    )
