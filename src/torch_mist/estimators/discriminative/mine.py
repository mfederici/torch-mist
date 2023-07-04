from typing import List, Dict, Any

from torch_mist.baselines import BatchLogMeanExp, ExponentialMovingAverage
from torch_mist.critic.base import Critic
from torch_mist.critic.utils import critic
from torch_mist.estimators.discriminative.tuba import TUBA


class MINE(TUBA):
    def __init__(
            self,
            critic: Critic,
            mc_samples: int = 1,
            gamma: float = 0.9,
    ):
        super().__init__(
            critic=critic,
            mc_samples=mc_samples,
            baseline=BatchLogMeanExp('all'),
            grad_baseline=ExponentialMovingAverage(gamma=gamma),
        )


def mine(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        mc_samples: int = 1,
        gamma: float = 0.9,
        critic_params: Dict[str, Any] = None,
) -> MINE:
    critic_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    return MINE(
        critic=critic_nn,
        mc_samples=mc_samples,
        gamma=gamma,
    )
