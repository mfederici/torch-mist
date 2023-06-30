from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import Baseline, baseline_nn
from torch_mist.critic import Critic, critic
from torch_mist.estimators.discriminative.nwj import NWJ


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

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:

        value_b = self.baseline.forward(f_, x, y)
        if value_b.ndim == 1:
            value_b = value_b.unsqueeze(1)
        assert value_b.ndim == f_.ndim

        return super()._compute_log_ratio(
            x=x,
            y=y,
            f=f - value_b,
            y_=y_,
            f_=f_ - value_b
        )

    def _compute_log_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> Optional[torch.Tensor]:

        if self.grad_baseline is None:
            return None

        grad_b = self.grad_baseline.forward(f_, x, y)
        if grad_b.ndim == 1:
            grad_b = grad_b.unsqueeze(1)
        assert grad_b.ndim == f_.ndim

        return super()._compute_log_ratio_grad(
            x=x,
            y=y,
            f=f - grad_b,
            y_=y_,
            f_=f_ - grad_b
        )

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
