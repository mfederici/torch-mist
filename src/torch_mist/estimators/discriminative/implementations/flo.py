from functools import lru_cache

import torch

from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    DiscriminativeMIEstimator,
)


class FLO(DiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        amortized_critic: Critic,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
        )
        self.amortized_critic = amortized_critic

    @lru_cache(maxsize=1)
    def amortized_estimate(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.amortized_critic(x=x, y=y)

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim
        # Approximate the log-ratio p(y|x)/p(y) on samples from p(x,y).
        # x and y have shape [..., X_DIM] and [..., Y_DIM] respectively

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x, y), with shape [...]
        f = self.unnormalized_log_ratio(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from r(x, y), with shape [M, ...]
        y_, w = self.sample_negatives(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)r(y|x)
        # The tensor f_ has shape [M, N...] in which f_[i,j] contains critic(x[j], y_[i,j]).
        # and y_ is sampled from r(y|x), which is set to the empirical p(y) unless a proposal is specified
        f_ = self.critic(x, y_)

        assert f_.shape[1:] == f.shape

        # Compute the negative amortized log ratio. It has shape [...]
        u = self.amortized_estimate(x=x, y=y)
        assert (
            u.ndim == f_.ndim - 1
        ), f"Baseline has ndim {u.ndim} while f_ has ndim {f_.ndim}"

        if not (w is None):
            assert f_.shape == w.shape
            f_ = f_ + w.log()

        loss = -u + (u - f + torch.logsumexp(f_, 0)).exp() / f_.shape[0] + 1

        return loss

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.amortized_estimate(x=x, y=y)

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  (critic): " + str(self.critic).replace("\n", "\n" + "  ") + "\n"
        )
        s += (
            "  (amortized_critic): "
            + str(self.amortized_critic).replace("\n", "\n" + "  ")
            + "\n"
        )
        s += "  (neg_samples): " + str(self.neg_samples) + "\n"
        s += ")"

        return s
