from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import LearnableJointBaseline, joint_baseline_nn
from torch_mist.critic.base import Critic
from torch_mist.critic.utils import critic
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator


class FLO(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            critic: Critic,
            baseline: LearnableJointBaseline,
            mc_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            mc_samples=mc_samples,
        )
        self.baseline = baseline

    def compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor,
    ) -> torch.Tensor:
        b = self.baseline(x=x, y=y, f_=f_)
        assert b.ndim == f.ndim

        M = f_.shape[0]

        log_ratio = -(b + (torch.logsumexp(f_, 0) - f - b).exp() / M) + 1

        return log_ratio

def flo(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        mc_samples: int = 1,
        critic_params: Dict[str, Any] = None,
        baseline_params: Dict[str, Any] = None,
) -> FLO:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    if baseline_params is None:
        baseline_params = {}
    baseline = joint_baseline_nn(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        **baseline_params
    )

    return FLO(
        critic=url_nn,
        baseline=baseline,
        mc_samples=mc_samples,
    )
