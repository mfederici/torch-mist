from typing import List

from pyro.nn import DenseNN
from torch import nn

from torch_mist.ratio import JointUnnormalizedRatioEstimator, SeparableUnnormalizedRatioEstimator


def separable_unnormalized_log_ratio(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        projection_head: str = 'symmetric',
        output_dim: int = 128,
        nonlinearity=nn.ReLU(True),
        temperature: float=0.1
):
    assert projection_head in ['symmetric', 'asymmetric', 'one'], \
        F"projection_heads must be one of ['symmetric', 'asymmetric', 'one'], got {projection_head}"

    if projection_head == 'one':
        output_dim = y_dim

    f_y = None

    assert len(hidden_dims) > 0
    f_x = DenseNN(
        input_dim=x_dim,
        hidden_dims=[int(h ** 0.5) for h in hidden_dims],
        param_dims=[output_dim],
        nonlinearity=nonlinearity,
    )

    if projection_head == 'symmetric':
        f_y = f_x
    elif projection_head == 'asymmetric':
        f_y = DenseNN(
            input_dim=y_dim,
            hidden_dims=[int(h ** 0.5) for h in hidden_dims],
            param_dims=[output_dim],
            nonlinearity=nonlinearity,
        )

    unnormalized_log_ratio = SeparableUnnormalizedRatioEstimator(
        f_x=f_x,
        f_y=f_y,
        temperature=temperature,
    )

    return unnormalized_log_ratio


def joint_unnormalized_log_ratio(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        nonlinearity=nn.ReLU(True),
):
    joint_net = DenseNN(
        input_dim=x_dim + y_dim,
        hidden_dims=hidden_dims,
        param_dims=[1],
        nonlinearity=nonlinearity,
    )
    unnormalized_log_ratio = JointUnnormalizedRatioEstimator(
        joint_net=joint_net,
    )

    return unnormalized_log_ratio


def unnormalized_log_ratio(x_dim, y_dim, hidden_dims, log_ratio_model, **kwargs):
    assert log_ratio_model in ['joint', 'separable'], \
        f'log_ratio_model must be one of [joint, separable, separable_asymm], got {log_ratio_model}'

    if log_ratio_model == 'joint':
        return joint_unnormalized_log_ratio(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
    else:
        return separable_unnormalized_log_ratio(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
