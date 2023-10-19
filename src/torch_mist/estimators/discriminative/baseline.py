import inspect
from typing import List, Dict, Any, Optional

import torch

from torch_mist.baselines import Baseline
from torch_mist.critic.base import Critic, CRITIC_TYPE, JOINT_CRITIC
from torch_mist.estimators.discriminative.base import (
    DiscriminativeMIEstimator,
)
from torch_mist.utils.caching import reset_cache_before_call


class BaselineDiscriminativeMIEstimator(DiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        baseline: Baseline,
        train_baseline: Optional[Baseline] = None,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
        )

        self.baseline = baseline
        self.train_baseline = train_baseline

    def log_normalization_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
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
        assert (
            b.ndim == f_.ndim - 1
        ), f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

        log_norm = (f_ - b.unsqueeze(0)).exp().mean(0) + b - 1.0

        return log_norm

    def compute_log_normalization_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        if self.train_baseline is None:
            return self.compute_log_normalization(x, y, f_)
        else:
            # Compute the gradient baseline. It has shape [...]
            b = self.train_baseline(f_, x, y)
            assert (
                b.ndim == f_.ndim - 1
            ), f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

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

        return -log_ratio.mean()

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  (ratio_estimator): "
            + str(self.critic).replace("\n", "\n" + "  ")
            + "\n"
        )
        s += (
            "  (baseline): "
            + str(self.baseline).replace("\n", "\n" + "  ")
            + "\n"
        )
        if self.train_baseline is not None:
            s += (
                "  (grad_baseline): "
                + str(self.train_baseline).replace("\n", "\n" + "  ")
                + "\n"
            )
        s += "  (neg_samples): " + str(self.neg_samples) + "\n"
        s += ")"

        return s
