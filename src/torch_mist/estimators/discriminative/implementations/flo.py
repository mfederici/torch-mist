from typing import Optional

import torch

from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    DiscriminativeMIEstimator,
)
from torch_mist.utils.caching import cached_method


class FLO(DiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        normalized_critic: Critic,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
        )
        self.normalized_critic = normalized_critic

    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.normalized_critic(x, y)

    @cached_method
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.normalized_critic(x, y)

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -super().log_ratio(x, y)

    def _approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.Tensor,
        log_w: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Evaluate the critic f on the samples x, y ~ p(x, y). It has shape [...]
        f = self.critic(x, y)

        # Compute the negative amortized log ratio u. It has shape [...]
        u = self.unnormalized_log_ratio(x=x, y=y)
        assert f.shape == u.shape

        # Add the log_weights if provided
        if not (log_w is None):
            assert log_w.ndim == f_.ndim
            f_ = f_ + log_w

        log_Z = (u - f + torch.logsumexp(f_, 0)).exp() / f_.shape[0] - 1

        assert log_Z.shape == f_.shape[1:]

        return log_Z

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  (critic): " + str(self.critic).replace("\n", "\n" + "  ") + "\n"
        )
        s += (
            "  (normalized_critic): "
            + str(self.normalized_critic).replace("\n", "\n" + "  ")
            + "\n"
        )
        s += "  (neg_samples): " + str(self.neg_samples) + "\n"
        s += ")"

        return s
