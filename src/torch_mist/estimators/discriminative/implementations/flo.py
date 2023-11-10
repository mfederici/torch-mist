import torch

from torch_mist.baseline import LearnableJointBaseline
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    DiscriminativeMIEstimator,
)
from torch_mist.utils.caching import cached


class FLO(DiscriminativeMIEstimator):
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
        f = self.unnormalized_log_ratio(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from r(x, y), with shape [M, ...]
        f_ = self.critic_on_negatives(x, y)

        assert f_.shape[1:] == f.shape

        # Compute the baseline. It has shape [...]
        b = self.baseline(f_, x, y)
        assert (
            b.ndim == f_.ndim - 1
        ), f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

        log_ratio = (
            -b - (torch.logsumexp(f_, 0) - f - b).exp() / f_.shape[0] + 1
        )

        return log_ratio
