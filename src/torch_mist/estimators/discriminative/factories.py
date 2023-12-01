import inspect
from typing import List

from torch_mist.baseline.base import ConstantBaseline
from torch_mist.baseline.factories import baseline_nn
from torch_mist.critic.base import JOINT_CRITIC, SEPARABLE_CRITIC
from torch_mist.critic.factories import critic_nn
from torch_mist.estimators.discriminative.implementations import (
    AlphaTUBA,
    FLO,
    InfoNCE,
    JS,
    MINE,
    NWJ,
    SMILE,
    TUBA,
)


def alpha_tuba(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    alpha: float = 0.01,
    learnable_baseline: bool = True,
    critic_type: str = JOINT_CRITIC,
    neg_samples: int = -1,
    **kwargs,
) -> AlphaTUBA:
    baseline_params = {}
    critic_params = {}

    for param_name, param_value in kwargs:
        if param_name in inspect.signature(baseline_nn).parameters:
            baseline_params[param_name] = param_value
        else:
            critic_params[param_name] = param_value
        if param_name in inspect.signature(critic_nn).parameters:
            critic_params[param_name] = param_value

    if learnable_baseline:
        b_nn = baseline_nn(
            x_dim=x_dim, hidden_dims=hidden_dims, **baseline_params
        )
    else:
        b_nn = ConstantBaseline(value=1.0)

    return AlphaTUBA(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **critic_params,
        ),
        baseline=b_nn,
        alpha=alpha,
        neg_samples=neg_samples,
    )


def flo(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    critic_type: str = JOINT_CRITIC,
    **kwargs,
) -> FLO:
    critic = critic_nn(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        **kwargs,
    )

    amortized_critic = critic_nn(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        **kwargs,
    )

    return FLO(
        critic=critic,
        amortized_critic=amortized_critic,
        neg_samples=neg_samples,
    )


def infonce(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    critic_type: str = SEPARABLE_CRITIC,
    **kwargs,
) -> InfoNCE:
    return InfoNCE(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            critic_type=critic_type,
            hidden_dims=hidden_dims,
            **kwargs,
        )
    )


def js(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    critic_type: str = JOINT_CRITIC,
    **kwargs,
) -> JS:
    return JS(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs,
        ),
        neg_samples=neg_samples,
    )


def mine(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    critic_type: str = JOINT_CRITIC,
    neg_samples: int = 1,
    gamma: float = 0.9,
    **kwargs,
) -> MINE:
    return MINE(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs,
        ),
        neg_samples=neg_samples,
        gamma=gamma,
    )


def nwj(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    critic_type: str = JOINT_CRITIC,
    **kwargs,
) -> NWJ:
    return NWJ(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs,
        ),
        neg_samples=neg_samples,
    )


def smile(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    tau: float = 5.0,
    critic_type: str = JOINT_CRITIC,
    **kwargs,
) -> SMILE:
    return SMILE(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs,
        ),
        neg_samples=neg_samples,
        tau=tau,
    )


def tuba(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    critic_type: str = JOINT_CRITIC,
    **kwargs,
) -> TUBA:
    baseline_params = {}
    critic_params = {}

    for param_name, param_value in kwargs:
        if param_name in inspect.signature(baseline_nn).parameters:
            baseline_params[param_name] = param_value
        else:
            critic_params[param_name] = param_value

    b_nn = baseline_nn(x_dim=x_dim, hidden_dims=hidden_dims, **baseline_params)

    return TUBA(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **critic_params,
        ),
        baseline=b_nn,
        neg_samples=neg_samples,
    )
