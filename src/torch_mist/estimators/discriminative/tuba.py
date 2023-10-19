import inspect
from typing import List

from torch_mist.baselines import LearnableBaseline
from torch_mist.critic.base import Critic, JOINT_CRITIC
from torch_mist.estimators.discriminative.baseline import (
    BaselineDiscriminativeMIEstimator,
)


class TUBA(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        baseline: LearnableBaseline,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            baseline=baseline,
            neg_samples=neg_samples,
        )


def tuba(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    critic_type: str = JOINT_CRITIC,
    **kwargs,
) -> TUBA:
    from torch_mist.baselines import baseline_nn
    from torch_mist.critic import critic_nn

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
