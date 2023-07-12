from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import Baseline
from torch_mist.critic.base import Critic
from torch_mist.estimators.discriminative.base import DiscriminativeMutualInformationEstimator
from torch_mist.utils.caching import reset_cache_before_call


class TUBA(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            critic: Critic,
            baseline: Baseline,
            grad_baseline: Optional[Baseline] = None,
            neg_samples: int = 1
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
        )

        self.baseline = baseline
        self.grad_baseline = grad_baseline

    def log_normalization_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from r(y|x), with shape [M, ...]
        f_ = self.critic_on_negatives(x, y)

        return self.compute_log_normalization_loss(x, y, f_)

    def compute_log_normalization(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f_: torch.Tensor,
    ) -> torch.Tensor:

        # Compute the baseline. It has shape [...]
        b = self.baseline(f_, x, y)
        assert b.ndim == f_.ndim-1, f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

        log_norm = (f_-b.unsqueeze(0)).exp().mean(0) + b - 1.0

        return log_norm

    def compute_log_normalization_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f_: torch.Tensor,
    ) -> torch.Tensor:
        if self.grad_baseline is None:
            return self.compute_log_normalization(x, y, f_)
        else:
            # Compute the gradient baseline. It has shape [...]
            b = self.grad_baseline(f_, x, y)
            assert b.ndim == f_.ndim - 1, f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

            log_norm = (f_ - b.unsqueeze(0)).exp().mean(0) - b + 1

            return log_norm

    @reset_cache_before_call
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim
        # Approximate the log-ratio p(y|x)/p(y) on samples from p(x,y).
        # x and y have shape [..., X_DIM] and [..., Y_DIM] respectively

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x, y), with shape [...]
        f = self.critic_on_positives(x, y)

        # Compute the log-normalization term, with shape [...]
        log_normalization = self.log_normalization_loss(x, y)

        assert log_normalization.ndim == f.ndim

        log_ratio = f - log_normalization
        assert log_ratio.ndim == y.ndim - 1

        return - log_ratio.mean()

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  (ratio_estimator): ' + str(self.critic).replace('\n', '\n' + '  ') + '\n'
        s += '  (baseline): ' + str(self.baseline).replace('\n', '\n' + '  ') + '\n'
        if self.grad_baseline is not None:
            s += '  (grad_baseline): ' + str(self.grad_baseline).replace('\n', '\n' + '  ') + '\n'
        s += '  (neg_samples): ' + str(self.neg_samples) + '\n'
        s += ')'

        return s


def tuba(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type: str = 'joint',
        neg_samples: int = 1,
        critic_params: Dict[str, Any] = None,
        baseline_params: Dict[str, Any] = None,
) -> TUBA:
    from torch_mist.baselines import baseline_nn
    from torch_mist.critic import critic_nn

    if baseline_params is None:
        baseline_params = {}

    b_nn = baseline_nn(
        x_dim=x_dim,
        hidden_dims=hidden_dims,
        **baseline_params
    )

    return TUBA(
        critic=critic_nn(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type=critic_type,
            critic_params=critic_params
        ),
        baseline=b_nn,
        neg_samples=neg_samples,
    )
