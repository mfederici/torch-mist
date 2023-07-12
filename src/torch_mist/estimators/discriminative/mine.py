from typing import List, Dict, Any

from torch_mist.baselines import BatchLogMeanExp, ExponentialMovingAverage
from torch_mist.critic.base import Critic
from torch_mist.estimators.discriminative.tuba import TUBA


class MINE(TUBA):
    def __init__(
            self,
            critic: Critic,
            neg_samples: int = 1,
            gamma: float = 0.9,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
            baseline=BatchLogMeanExp('all'),
            grad_baseline=ExponentialMovingAverage(gamma=gamma),
        )


def mine(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        neg_samples: int = 1,
        gamma: float = 0.9,
        critic_params: Dict[str, Any] = None,
) -> MINE:
    from torch_mist.critic.utils import critic_nn

    return MINE(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            critic_params=critic_params
        ),
        neg_samples=neg_samples,
        gamma=gamma,
    )
