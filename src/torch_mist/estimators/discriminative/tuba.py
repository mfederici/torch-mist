from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import Baseline, baseline_nn
from torch_mist.critic.base import Critic
from torch_mist.critic.utils import critic
from torch_mist.estimators.discriminative.nwj import NWJ
from torch_mist.utils.caching import reset_cache_before_call


class TUBA(NWJ):
    def __init__(
            self,
            critic: Critic,
            baseline: Baseline,
            grad_baseline: Optional[Baseline] = None,
            mc_samples: int = 1
    ):

        super().__init__(
            critic=critic,
            mc_samples=mc_samples,
        )

        self.baseline = baseline
        self.grad_baseline = grad_baseline

    def compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor,
    ) -> torch.Tensor:

        b = self.baseline(f_, x, y)
        assert b.ndim == f.ndim, f"Baseline output has shape {b.shape} but should have shape {f.shape}"

        return super().compute_log_ratio(
            x=x,
            y=y,
            f=f - b,
            f_=f_ - b.unsqueeze(0)
        )

    @reset_cache_before_call
    def loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.grad_baseline is None:
            return super().loss(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)p(y|x), with shape [N]
        f = self.critic_on_positives(x, y)
        assert f.shape == y.shape[:-1]

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)r(y|x), with shape [N, M']
        f_ = self.critic_on_negatives(x, y)

        grad_b = self.grad_baseline.forward(f_, x, y)
        assert grad_b.shape == f.shape

        return -super().compute_log_ratio(
            x=x,
            y=y,
            f=f - grad_b,
            f_=f_ - grad_b.unsqueeze(0)
        ).mean()

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  (ratio_estimator): ' + str(self.critic).replace('\n', '\n' + '  ') + '\n'
        s += '  (baseline): ' + str(self.baseline).replace('\n', '\n' + '  ') + '\n'
        if self.grad_baseline is not None:
            s += '  (grad_baseline): ' + str(self.grad_baseline).replace('\n', '\n' + '  ') + '\n'
        s += '  (mc_samples): ' + str(self.mc_samples) + '\n'
        s += ')'

        return s


def tuba(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        mc_samples: int = 1,
        critic_params: Dict[str, Any] = None,
        baseline_params: Dict[str, Any] = None,
) -> TUBA:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        critic_params=critic_params
    )

    if baseline_params is None:
        baseline_params = {}

    b_nn = baseline_nn(
        x_dim=x_dim,
        hidden_dims=hidden_dims,
        **baseline_params
    )

    return TUBA(
        critic=url_nn,
        baseline=b_nn,
        mc_samples=mc_samples,
    )
