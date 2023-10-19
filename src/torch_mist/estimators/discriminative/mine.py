from typing import List, Dict, Any

from torch_mist.baselines import BatchLogMeanExp, ExponentialMovingAverage
from torch_mist.critic.base import Critic, JOINT_CRITIC, CRITIC_TYPE
from torch_mist.estimators.discriminative.baseline import (
    BaselineDiscriminativeMIEstimator,
)


class MINE(BaselineDiscriminativeMIEstimator):
    lower_bound = False  # Technically MINE is a lower bound but sometimes it converges from above

    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
        gamma: float = 0.9,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
            baseline=BatchLogMeanExp("all"),
            train_baseline=ExponentialMovingAverage(gamma=gamma),
        )


def mine(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    critic_type: str = JOINT_CRITIC,
    neg_samples: int = 1,
    gamma: float = 0.9,
    **kwargs
) -> MINE:
    from torch_mist.critic.utils import critic_nn

    return MINE(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            **kwargs
        ),
        neg_samples=neg_samples,
        gamma=gamma,
    )
