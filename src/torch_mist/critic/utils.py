from typing import List

from pyro.nn import DenseNN
from torch import nn

from torch_mist.critic import JointCritic, SeparableCritic


def separable_critic(
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

    unnormalized_log_ratio = SeparableCritic(
        f_x=f_x,
        f_y=f_y,
        temperature=temperature,
    )

    return unnormalized_log_ratio


def joint_critic(
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
    unnormalized_log_ratio = JointCritic(
        joint_net=joint_net,
    )

    return unnormalized_log_ratio


def critic(x_dim, y_dim, hidden_dims, critic_type, **kwargs):
    assert critic_type in ['joint', 'separable'], \
        f'critic must be one of [joint, separable, separable_asymm], got {critic_type}'

    if critic_type == 'joint':
        return joint_critic(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
    else:
        return separable_critic(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
