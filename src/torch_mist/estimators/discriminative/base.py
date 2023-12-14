from abc import abstractmethod
from typing import Union, Dict
from functools import lru_cache

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch_mist.baseline import Baseline
from torch_mist.distributions.empirical import EmpiricalDistribution
from torch_mist.estimators.base import MIEstimator
from torch_mist.critic import Critic
from torch_mist.critic import SeparableCritic
from torch_mist.utils.indexing import select_off_diagonal


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
        self._proposal = EmpiricalDistribution()

    def set_proposal(
        self, proposal: Union[Distribution, ConditionalDistribution]
    ):
        self._proposal = proposal

    @lru_cache(maxsize=1)
    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        f = self.critic(x, y)
        assert f.ndim == y.ndim - 1
        return f

    def get_n_negatives(self, N: int):
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

    @lru_cache(maxsize=1)
    def critic_on_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self._proposal, EmpiricalDistribution):
            assert y is not None
            self._proposal.add_samples(y)

        N = x.shape[0]
        neg_samples = self.get_n_negatives(N)

        # If we are sampling using the empirical distribution and the critic is separable
        # We can use an efficient implementation
        if isinstance(self._proposal, Distribution) and isinstance(
            self.critic, SeparableCritic
        ):
            # Efficient implementation for separable critic with empirical distribution (negatives from the same batch)
            if isinstance(self._proposal, EmpiricalDistribution):
                # Keep only neg_samples negatives for each positive (off-diagonal)
                if neg_samples < N:
                    # Override the default self.critic(x,y) for efficient computation
                    f_x = self.critic.f_x(x)
                    f_y = self.critic.f_y(y)
                    off_diagonal_f_y = select_off_diagonal(f_y, neg_samples)
                    f_ = torch.einsum(
                        "ab...c, b...c -> ab...", off_diagonal_f_y, f_x
                    )
                else:
                    # Take the M stored samples from y (from the empirical). Shape [M, 1, ...,Y_DIM]
                    y_ = self._proposal._samples[:neg_samples].unsqueeze(1)
                    # Compute the critic on all pairs
                    f_ = self.critic(x, y_)
            else:
                # Efficient implementation for separable critic with unconditional proposal
                # Here we re-use the same samples y'~ r(y) for all the batch
                # Sample from the proposal r(y) [M', 1, ..., Y_DIM] with M' as the number of neg_samples
                y_ = self._proposal.sample(
                    sample_shape=torch.Size([neg_samples])
                ).unsqueeze(1)

                # Compute the critic on them. Shape [M', ...]
                f_ = self.critic(x, y_)
        else:
            # Sample from the proposal r(y|x) [M, ..., Y_DIM] with M as the number of neg_samples
            if isinstance(self._proposal, ConditionalDistribution):
                proposal = self._proposal.condition(x)
            else:
                proposal = self._proposal

            y_ = proposal.sample(sample_shape=torch.Size([neg_samples]))
            # The shape of the samples from the proposal distribution is [M, ..., Y_DIM]
            assert y_.ndim == x.ndim + 1 and y_.shape[0] == neg_samples
            assert y_.shape[0] == neg_samples and y_.ndim == x.ndim + 1

            # Compute the log-ratio on the samples from the proposal p(x)r(y|x) [M,...]
            f_ = self.critic(x, y_)

        assert (
            f_.shape[0] == neg_samples
        ), f"Expected {neg_samples} samples, got {f_.shape[0]}"
        assert (
            f_.shape[1:] == x.shape[:-1]
        ), f"Negatives have shape {f_.shape} shape, while x has shape {x.shape}"

        if isinstance(self._proposal, EmpiricalDistribution):
            self._proposal.update()

        return f_

    def approx_log_partition(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)r(y|x)
        # The tensor f_ has shape [M, N...] in which f_[i,j] contains critic(x[j], y_[i,j]).
        # and y_ is sampled from r(y|x), which is set to the empirical p(y) unless a proposal is specified
        f_ = self.critic_on_negatives(x, y)

        log_Z = self.batch_approx_log_partition(x, y, f_)

        N = x.shape[0]
        assert log_Z.shape[0] == self.get_n_negatives(N)
        assert (
            not isinstance(x, torch.LongTensor)
            and log_Z.shape[1:] == x.shape[:-1]
        ) or (isinstance(x, torch.LongTensor) and log_Z.shape[1:] == x.shape)

        return log_Z.mean(0)

    @abstractmethod
    def batch_approx_log_partition(
        self, x: torch.Tensor, y: torch.Tensor, f_: torch.Tensor
    ):
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

    def set_proposal(
        self, proposal: Union[Distribution, ConditionalDistribution]
    ):
        self.train_estimator.set_proposal(proposal)
        self.eval_estimator.set_proposal(proposal)

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.train_estimator.batch_loss(x, y)

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.eval_estimator.log_ratio(x, y)


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

    def batch_approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        # Compute the baseline. It has shape [1,...]
        b = self.baseline(f_, x, y).unsqueeze(0)
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
