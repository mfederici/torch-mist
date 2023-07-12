from typing import List, Dict, Any

from torch_mist.estimators.discriminative.tuba import TUBA
from torch_mist.critic.base import Critic
from torch_mist.baselines import Baseline, InterpolatedBaseline, BatchLogMeanExp


class AlphaTUBA(TUBA):
    def __init__(
            self,
            critic: Critic,
            baseline: Baseline,
            alpha: float = 0.5,
            neg_samples: int = -1,
    ):
        alpha_baseline = InterpolatedBaseline(
            baseline_1=BatchLogMeanExp('first'),
            baseline_2=baseline,
            alpha=alpha
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
        critic_type: str = 'joint',
        alpha: float = 0.01,
        learnable_baseline: bool = True,
        neg_samples=-1,
        critic_params: Dict[str, Any] = None,
        baseline_params: Dict[str, Any] = None,
) -> AlphaTUBA:
    from torch_mist.critic.utils import critic_nn
    from torch_mist.baselines import ConstantBaseline, baseline_nn

    if learnable_baseline:
        if baseline_params is None:
            baseline_params = {}
        b_nn = baseline_nn(
            x_dim=x_dim,
            hidden_dims=hidden_dims,
            **baseline_params
        )
    else:
        assert baseline_params is None
        b_nn = ConstantBaseline(value=1.0)

    return AlphaTUBA(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            critic_params=critic_params
        ),
        baseline=b_nn,
        alpha=alpha,
        neg_samples=neg_samples,
    )
