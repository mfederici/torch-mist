from abc import abstractmethod
from typing import Optional, Union

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch_mist.estimators.base import MIEstimator
from torch_mist.critic.base import Critic
from torch_mist.critic.separable import SeparableCritic
from torch_mist.utils.caching import (
    cached,
    reset_cache_before_call,
    reset_cache_after_call,
)
from torch_mist.utils.indexing import select_off_diagonal


class EmpiricalDistribution(Distribution):
    def __init__(self):
        super().__init__(validate_args=False)
        self._samples = None

    def add_samples(self, samples):
        self._samples = samples

    def sample(self, sample_shape=torch.Size()):
        assert self._samples is not None
        assert len(sample_shape) == 1
        n_samples = sample_shape[0]

        return select_off_diagonal(self._samples, n_samples)

    def update(self):
        self._samples = None


class DiscriminativeMIEstimator(MIEstimator):
    lower_bound: bool = True

    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
        proposal: Optional[
            Union[Distribution, ConditionalDistribution]
        ] = None,
    ):
        super().__init__()
        self.critic = critic
        self.neg_samples = neg_samples
        if proposal is None:
            proposal = EmpiricalDistribution()
        self.proposal = proposal

    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.critic_on_positives(x, y)

    @cached
    def critic_on_positives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        f = self.critic(x, y)
        assert f.ndim == y.ndim - 1
        return f

    @cached
    def critic_on_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.proposal, EmpiricalDistribution):
            assert y is not None
            self.proposal.add_samples(y)

        neg_samples = self.neg_samples
        N = x.shape[0]

        # Negative neg_samples values are interpreted as difference from the batch size (-1 is all but one in the batch)
        if neg_samples <= 0:
            neg_samples = N + neg_samples

        # We can't use more negative than the batch
        if neg_samples > N:
            neg_samples = N

        # If we are sampling using the empirical distribution and the critic is separable
        # We can use an efficient implementation
        if isinstance(self.proposal, Distribution) and isinstance(
            self.critic, SeparableCritic
        ):
            # Efficient implementation for separable critic with empirical distribution (negatives from the same batch)
            if isinstance(self.proposal, EmpiricalDistribution):
                if neg_samples != N:
                    f_ = self.critic(x, y, neg_samples)
                else:
                    # Take the N stored samples from y (stored in _samples). Shape [N, 1, ...,Y_DIM]
                    y_ = self.proposal._samples[:N].unsqueeze(1)

                    # Compute the critic on them. Shape [N, N, ...]
                    f_ = self.critic(x, y_)

            else:
                # Efficient implementation for separable critic with unconditional proposal
                # Here we re-use the same samples y'~ r(y) for all the batch
                # Sample from the proposal r(y) [M', 1, ..., Y_DIM] with M' as the number of neg_samples
                y_ = self.proposal.sample(
                    sample_shape=torch.Size([neg_samples])
                ).unsqueeze(1)

                # Compute the critic on them. Shape [M', ...]
                f_ = self.critic(x, y_)
        else:
            # Sample from the proposal r(y|x) [M, ..., Y_DIM] with M as the number of neg_samples
            if isinstance(self.proposal, ConditionalDistribution):
                proposal = self.proposal.condition(x)
            else:
                proposal = self.proposal

            y_ = proposal.sample(sample_shape=torch.Size([neg_samples]))
            # The shape of the samples from the proposal distribution is [M, ..., Y_DIM]
            assert y_.ndim == x.ndim + 1 and y_.shape[0] == neg_samples
            assert y_.shape[0] == neg_samples and y_.ndim == x.ndim + 1

            # Compute the log-ratio on the samples from the proposal p(x)r(y|x) [M,...]
            f_ = self.critic(x, y_)

        assert (
            f_.ndim == x.ndim
        ), f"Expected {x.ndim} dimensions, got {f_.ndim}"
        assert (
            f_.shape[0] == neg_samples
        ), f"Expected {neg_samples} samples, got {f_.shape[0]}"

        if isinstance(self.proposal, EmpiricalDistribution):
            self.proposal.update()

        return f_

    def log_normalization(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from r(y|x), with shape [M, ...]
        f_ = self.critic_on_negatives(x, y)

        return self.compute_log_normalization(x, y, f_)

    @abstractmethod
    def compute_log_normalization(
        self, x: torch.Tensor, y: torch.Tensor, f_: torch.Tensor
    ):
        raise NotImplementedError()

    @reset_cache_after_call
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim
        # Approximate the log-ratio p(y|x)/p(y) on samples from p(x,y).
        # x and y have shape [..., X_DIM] and [..., Y_DIM] respectively

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x, y), with shape [...]
        f = self.critic_on_positives(x, y)

        # Compute the log-normalization term, with shape [...]
        log_normalization = self.log_normalization(x, y)

        assert log_normalization.ndim == f.ndim

        log_ratio = f - log_normalization
        assert log_ratio.ndim == y.ndim - 1

        return log_ratio

    @reset_cache_before_call
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.log_ratio(x, y).mean()

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  (critic): " + str(self.critic).replace("\n", "\n" + "  ") + "\n"
        )
        s += "  (neg_samples): " + str(self.neg_samples) + "\n"
        s += ")"

        return s
