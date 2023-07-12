from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import LearnableJointBaseline
from torch_mist.critic.base import Critic
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator
from torch_mist.utils.caching import cached


class FLO(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            critic: Critic,
            baseline: LearnableJointBaseline,
            neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
        )
        self.baseline = baseline

    @cached
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim
        # Approximate the log-ratio p(y|x)/p(y) on samples from p(x,y).
        # x and y have shape [..., X_DIM] and [..., Y_DIM] respectively

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x, y), with shape [...]
        f = self.critic_on_positives(x, y)

        f_ = self.critic_on_negatives(x, y)
        assert f_.ndim == f.ndim + 1

        # Compute the baseline. It has shape [...]
        b = self.baseline(f_, x, y)
        assert b.ndim == f_.ndim-1, f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

        log_ratio = - b - (torch.logsumexp(f_, 0)-f-b).exp()/f_.shape[0] + 1

        return log_ratio


def flo(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        neg_samples: int = 1,
        critic_params: Dict[str, Any] = None,
        baseline_params: Dict[str, Any] = None,
) -> FLO:
    from torch_mist.critic.utils import critic_nn
    from torch_mist.baselines import joint_baseline_nn

    if baseline_params is None:
        baseline_params = {}
    baseline = joint_baseline_nn(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        **baseline_params
    )

    return FLO(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            critic_params=critic_params
        ),
        baseline=baseline,
        neg_samples=neg_samples,
    )
