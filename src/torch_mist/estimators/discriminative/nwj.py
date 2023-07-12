from typing import List, Dict, Any

from torch_mist.estimators.discriminative.tuba import TUBA
from torch_mist.critic.base import Critic
from torch_mist.baselines import ConstantBaseline


class NWJ(TUBA):
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
        critic_type: str = 'joint',
        critic_params: Dict[str, Any] = None,
) -> NWJ:
    from torch_mist.critic.utils import critic_nn

    return NWJ(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            critic_params=critic_params
        ),
        neg_samples=neg_samples,
    )
