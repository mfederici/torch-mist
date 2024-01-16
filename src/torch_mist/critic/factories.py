from typing import List, Optional, Tuple

from torch import nn

from torch_mist.critic.separable import SeparableCritic
from torch_mist.critic.joint import JointCritic
from torch_mist.critic.base import CRITIC_TYPES, JOINT_CRITIC, Critic
from torch_mist.nn import Normalize, multi_head_dense_nn

SYMMETRIC_HEADS = "symmetric"
ASYMMETRIC_HEADS = "asymmetric"
ONE_HEAD = "one"
POSSIBLE_HEADS = [SYMMETRIC_HEADS, ASYMMETRIC_HEADS, ONE_HEAD]


def shared_joint_critics(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    nonlinearity: nn.Module = nn.ReLU(True),
    n_shared_layers: int = -1,
    n_critics: int = 2,
    **kwargs,
) -> Tuple[JointCritic, ...]:
    if len(kwargs) > 0:
        print(f"[Warning]: Parameters {kwargs} have been ignored")

    critic_nets = multi_head_dense_nn(
        input_dim=x_dim + y_dim,
        output_dim=1,
        hidden_dims=hidden_dims,
        n_shared_layers=n_shared_layers,
        n_heads=n_critics,
        cached_shared_forward=True,
        nonlinearity=nonlinearity,
    )
    critics = [JointCritic(critic_net) for critic_net in critic_nets]
    return tuple(critics)


def shared_separable_critics(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    projection_head: str = ASYMMETRIC_HEADS,
    nonlinearity: nn.Module = nn.ReLU(True),
    normalize: bool = False,
    temperature: float = 1.0,
    n_critics: int = 2,
    n_shared_layers: int = -1,
    k_dim: Optional[int] = None,
    **kwargs,
) -> Tuple[SeparableCritic, ...]:
    if len(kwargs) > 0:
        print(f"[Warning]: Parameters {kwargs} have been ignored")

    if not (projection_head in POSSIBLE_HEADS):
        raise ValueError(
            f"projection_heads must be one of {POSSIBLE_HEADS}, got {projection_head}"
        )

    if projection_head == ONE_HEAD:
        output_dim = y_dim
    elif k_dim:
        output_dim = k_dim
    else:
        assert (
            len(hidden_dims) > 0
        ), "At least one hidden dimension needs to be specified"
        output_dim = hidden_dims[-1]
        hidden_dims = hidden_dims[:-1]

    f_xs = multi_head_dense_nn(
        input_dim=x_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        nonlinearity=nonlinearity,
        n_heads=n_critics,
        cached_shared_forward=True,
        n_shared_layers=n_shared_layers,
    )

    if projection_head == SYMMETRIC_HEADS:
        assert x_dim == y_dim
        f_ys = f_xs
    elif projection_head == ASYMMETRIC_HEADS:
        f_ys = multi_head_dense_nn(
            input_dim=y_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            nonlinearity=nonlinearity,
            n_heads=2,
            cached_shared_forward=True,
            n_shared_layers=n_shared_layers,
        )
    else:
        f_ys = (None, None)

    if normalize:
        f_xs = [nn.Sequential(f_x, Normalize()) for f_x in f_xs]
        f_ys = [
            nn.Sequential(f_y, Normalize()) if f_y else Normalize()
            for f_y in f_ys
        ]

    critics = [
        SeparableCritic(
            f_x=f_xs[i],
            f_y=f_ys[i],
            temperature=temperature,
        )
        for i in range(n_critics)
    ]

    return tuple(critics)


def shared_critic_nns(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    critic_type: str,
    n_critics: int,
    n_shared_layers: int = -1,
    **kwargs,
) -> Tuple[Critic, ...]:
    if not (critic_type in CRITIC_TYPES):
        raise ValueError(
            f"critic must be one of {CRITIC_TYPES}, got {critic_type}"
        )

    if critic_type == JOINT_CRITIC:
        return shared_joint_critics(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            n_critics=n_critics,
            n_shared_layers=n_shared_layers,
            **kwargs,
        )
    else:
        return shared_separable_critics(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            n_critics=n_critics,
            n_shared_layers=n_shared_layers,
            **kwargs,
        )


def separable_critic(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    projection_head: str = ASYMMETRIC_HEADS,
    nonlinearity: nn.Module = nn.ReLU(True),
    normalize: bool = False,
    temperature: float = 1.0,
    k_dim: Optional[int] = None,
) -> SeparableCritic:
    return shared_separable_critics(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        projection_head=projection_head,
        nonlinearity=nonlinearity,
        normalize=normalize,
        temperature=temperature,
        k_dim=k_dim,
        n_critics=1,
        n_shared_layers=0,
    )[0]


def joint_critic(
    x_dim: int,
    y_dim: int,
    hidden_dims: Optional[List[int]],
    nonlinearity: nn.Module = nn.ReLU(True),
) -> JointCritic:
    return shared_joint_critics(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        nonlinearity=nonlinearity,
        n_critics=1,
        n_shared_layers=0,
    )[0]


def critic_nn(
    x_dim: int, y_dim: int, hidden_dims: List[int], critic_type: str, **kwargs
) -> Critic:
    return shared_critic_nns(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        n_critics=1,
        n_shared_layers=0,
        **kwargs,
    )[0]
