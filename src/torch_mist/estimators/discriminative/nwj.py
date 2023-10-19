from typing import List, Dict, Any

from torch_mist.estimators.discriminative.baseline import (
    BaselineDiscriminativeMIEstimator,
)
from torch_mist.critic.base import Critic, CRITIC_TYPE, JOINT_CRITIC
from torch_mist.baselines import ConstantBaseline


class NWJ(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
            baseline=ConstantBaseline(value=1.0),
        )


def nwj(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    neg_samples: int = 1,
    critic_type: str = JOINT_CRITIC,
    **kwargs
) -> NWJ:
    from torch_mist.critic.utils import critic_nn

    return NWJ(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs
        ),
        neg_samples=neg_samples,
    )
