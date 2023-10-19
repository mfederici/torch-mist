import inspect
from typing import List

from torch_mist.critic.base import Critic, JOINT_CRITIC
from torch_mist.baselines import (
    LearnableBaseline,
    InterpolatedBaseline,
    BatchLogMeanExp,
)
from torch_mist.estimators.discriminative.baseline import (
    BaselineDiscriminativeMIEstimator,
)


class AlphaTUBA(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        baseline: LearnableBaseline,
        alpha: float = 0.5,
        neg_samples: int = -1,
    ):
        alpha_baseline = InterpolatedBaseline(
            baseline_1=BatchLogMeanExp("first"),
            baseline_2=baseline,
            alpha=alpha,
        )

        super().__init__(
            critic=critic,
            baseline=alpha_baseline,
            neg_samples=neg_samples,
        )


def alpha_tuba(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    alpha: float = 0.01,
    learnable_baseline: bool = True,
    critic_type: str = JOINT_CRITIC,
    neg_samples: int = -1,
    **kwargs
) -> AlphaTUBA:
    from torch_mist.critic.utils import critic_nn
    from torch_mist.baselines import ConstantBaseline, baseline_nn

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
            **critic_params
        ),
        baseline=b_nn,
        alpha=alpha,
        neg_samples=neg_samples,
    )
