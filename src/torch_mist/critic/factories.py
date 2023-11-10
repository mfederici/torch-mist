from typing import List, Optional

from pyro.nn import DenseNN
from torch import nn

from torch_mist.critic.separable import SeparableCritic
from torch_mist.critic.joint import JointCritic
from torch_mist.critic.base import CRITIC_TYPES, JOINT_CRITIC
from torch_mist.nn import Normalize

SYMMETRIC_HEADS = "symmetric"
ASYMMETRIC_HEADS = "asymmetric"
ONE_HEAD = "one"
POSSIBLE_HEADS = [SYMMETRIC_HEADS, ASYMMETRIC_HEADS, ONE_HEAD]


def separable_critic(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    projection_head: str = ASYMMETRIC_HEADS,
    output_dim: Optional[int] = None,
    nonlinearity=nn.ReLU(True),
    normalize: bool = False,
    temperature: float = 1.0,
):
    if not (projection_head in POSSIBLE_HEADS):
        raise ValueError(
            f"projection_heads must be one of {POSSIBLE_HEADS}, got {projection_head}"
        )

    if projection_head == ONE_HEAD:
        output_dim = y_dim
    else:
        if output_dim is None:
            raise ValueError("output_dim must be specified.")

    f_y = None

    assert len(hidden_dims) > 0
    f_x = DenseNN(
        input_dim=x_dim,
        hidden_dims=hidden_dims,
        param_dims=[output_dim],
        nonlinearity=nonlinearity,
    )
    if normalize:
        f_x = nn.Sequential(f_x, Normalize())

    if projection_head == SYMMETRIC_HEADS:
        f_y = f_x
    elif projection_head == ASYMMETRIC_HEADS:
        f_y = DenseNN(
            input_dim=y_dim,
            hidden_dims=hidden_dims,
            param_dims=[output_dim],
            nonlinearity=nonlinearity,
        )
        if normalize:
            f_y = nn.Sequential(f_y, Normalize())

    critic = SeparableCritic(
        f_x=f_x,
        f_y=f_y,
        temperature=temperature,
    )

    return critic


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


def critic_nn(
    x_dim: int, y_dim: int, hidden_dims: List[int], critic_type: str, **kwargs
):
    if not (critic_type in CRITIC_TYPES):
        raise ValueError(
            f"critic must be one of {CRITIC_TYPES}, got {critic_type}"
        )

    if critic_type == JOINT_CRITIC:
        return joint_critic(
            x_dim=x_dim, y_dim=y_dim, hidden_dims=hidden_dims, **kwargs
        )
    else:
        return separable_critic(
            x_dim=x_dim, y_dim=y_dim, hidden_dims=hidden_dims, **kwargs
        )
