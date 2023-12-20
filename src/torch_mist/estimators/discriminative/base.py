from abc import abstractmethod
from contextlib import contextmanager
from typing import Dict, Tuple, Optional
from functools import lru_cache

import torch
from torch.distributions import Distribution

from torch_mist.baseline import Baseline
from torch_mist.estimators.base import MIEstimator
from torch_mist.critic import Critic
from torch_mist.critic import SeparableCritic
from torch_mist.estimators.discriminative.utils import SampleBuffer


class DiscriminativeMIEstimator(MIEstimator):
    lower_bound: bool = True
    infomax_gradient: Dict[str, bool] = {"x": True, "y": True}

    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
    ):
        super().__init__()
        self.critic = critic
        self.neg_samples = neg_samples
        self._y_buffer = SampleBuffer()

    @contextmanager
    def use_critic(self, critic: Critic):
        # Save the original value
        original_critic = self.critic
        # Set the new temporary value
        self.critic = critic
        try:
            yield
        finally:
            # Revert to the original value
            self.critic = original_critic

    @lru_cache(maxsize=1)
    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        f = self.critic(x, y)
        assert f.ndim == y.ndim - 1
        return f

    def n_negatives_to_use(self, N: int):
        neg_samples = self.neg_samples

        # Negative neg_samples values are interpreted as difference from the batch size (-1 is all but one in the batch)
        if neg_samples <= 0:
            neg_samples = N + neg_samples

        # We can't use more negative than the batch
        if neg_samples > N:
            neg_samples = N

        # At least one negative sample
        neg_samples = max(neg_samples, 1)
        return neg_samples

    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        N = x.shape[0]
        neg_samples = self.n_negatives_to_use(N)

        self._y_buffer.add_samples(y)

        if isinstance(self.critic, SeparableCritic) and neg_samples == N:
            # Efficient implementation for separable critic with empirical distribution (negatives from the same batch)
            y_ = self._y_buffer._samples[:N].unsqueeze(1)
        else:
            # Sample from the proposal p(y) [M, ..., Y_DIM] with M as the number of neg_samples
            y_ = self._y_buffer.sample(neg_samples)
            # The shape of the samples from the proposal distribution is [M, ..., Y_DIM]
            assert y_.ndim == x.ndim + 1 and y_.shape[0] == neg_samples
            assert y_.shape[0] == neg_samples and y_.ndim == x.ndim + 1

        self._y_buffer.update()

        return y_, None

    @lru_cache(maxsize=1)
    def approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        y_, w = self.sample_negatives(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)r(y|x)
        # The tensor f_ has shape [M, N...] in which f_[i,j] contains critic(x[j], y_[i,j]).
        # and y_ is sampled from r(y|x), which is set to the empirical p(y) unless a proposal is specified
        f_ = self.critic(x, y_)

        log_Z = self._approx_log_partition(x, f_)

        if not (w is None):
            assert w.shape == log_Z.shape
            log_Z = w * log_Z

        assert log_Z.shape[0] == self.n_negatives_to_use(x.shape[0])
        assert (
            not isinstance(x, torch.LongTensor)
            and log_Z.shape[1:] == x.shape[:-1]
        ) or (isinstance(x, torch.LongTensor) and log_Z.shape[1:] == x.shape)

        return log_Z.mean(0)

    @abstractmethod
    def _approx_log_partition(
        self, x: torch.Tensor, f_: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim
        # Approximate the log-ratio p(y|x)/p(y) on samples from p(x,y).
        # x and y have shape [..., X_DIM] and [..., Y_DIM] respectively

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x, y), with shape [...]
        unnormalized_log_ratio = self.unnormalized_log_ratio(x, y)

        # Compute the log-normalization term, with shape [M, ...]
        log_partition = self.approx_log_partition(x, y)

        log_ratio = unnormalized_log_ratio - log_partition
        assert log_ratio.ndim == y.ndim - 1

        return log_ratio

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_ratio(x, y)

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  (critic): " + str(self.critic).replace("\n", "\n" + "  ") + "\n"
        )
        s += "  (neg_samples): " + str(self.neg_samples) + "\n"
        s += ")"

        return s


class CombinedDiscriminativeMIEstimator(DiscriminativeMIEstimator):
    def __init__(
        self,
        train_estimator: DiscriminativeMIEstimator,
        eval_estimator: DiscriminativeMIEstimator,
    ):
        assert train_estimator.critic == eval_estimator.critic
        assert train_estimator.neg_samples == eval_estimator.neg_samples

        super().__init__(
            critic=train_estimator.critic,
            neg_samples=train_estimator.neg_samples,
        )

        self.train_estimator = train_estimator
        self.eval_estimator = eval_estimator
        self.infomax_gradient = train_estimator.infomax_gradient

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.train_estimator.batch_loss(x, y)

    def _approx_log_partition(
        self, x: torch.Tensor, f_: torch.Tensor
    ) -> torch.Tensor:
        return self.eval_estimator._approx_log_partition(x, f_)


class BaselineDiscriminativeMIEstimator(DiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        baseline: Baseline,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
        )

        self.baseline = baseline

    def _approx_log_partition(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the baseline. It has shape [1,...]
        b = self.baseline(f_, x).unsqueeze(0)
        assert (
            b.ndim == f_.ndim
        ), f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

        log_norm = (f_ - b).exp() + b - 1.0

        return log_norm

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
        s += "  (neg_samples): " + str(self.neg_samples) + "\n"
        s += ")"

        return s
