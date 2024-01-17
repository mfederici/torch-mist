from abc import abstractmethod
from typing import Dict, Tuple, Optional

import torch

from torch_mist.baseline import Baseline
from torch_mist.estimators.base import MIEstimator
from torch_mist.critic import Critic
from torch_mist.critic import SeparableCritic
from torch_mist.distributions.empirical import EmpiricalDistribution
from torch_mist.utils.caching import cached_method
from torch_mist.utils.indexing import matrix_off_diagonal


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
        self.proposal = EmpiricalDistribution()

    @cached_method
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

    @cached_method
    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        N = x.shape[0]
        neg_samples = self.n_negatives_to_use(N)

        if isinstance(self.proposal, EmpiricalDistribution):
            self.proposal.add_samples(y)

        # Efficient implementation for separable critic with empirical distribution (negatives from the same batch)
        if (
            isinstance(self.critic, SeparableCritic)
            and self.neg_samples <= 0
            and isinstance(self.proposal, EmpiricalDistribution)
        ):
            y_ = self.proposal._samples[:N].unsqueeze(1)
        else:
            # Sample from the proposal p(y) [M, ..., Y_DIM] with M as the number of neg_samples
            y_ = self.proposal.sample(torch.Size([neg_samples]))
            # The shape of the samples from the proposal distribution is [M, ..., Y_DIM]
            assert y_.ndim == x.ndim + 1 and y_.shape[0] == neg_samples
            assert y_.shape[0] == neg_samples and y_.ndim == x.ndim + 1

        if isinstance(self.proposal, EmpiricalDistribution):
            self.proposal.update()

        return x.unsqueeze(0), y_, None

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Approximate the log-ratio p(x,y)/r(x,y) on samples from p(x,y).
        # x and y have shape [..., X_DIM] and [..., Y_DIM] respectively

        assert x.ndim == y.ndim
        # Sample x_ and y_ from r(x,y). w represents an optional log-weighting coefficient log[r(x,y)/q(x,y)] in case
        # x_ and y_ are sampled from q(x,y) instead of r(x,y).
        # By default we have r(x,y) = p(x)p(y) and shape [M, ..., X_DIM] and [M, ..., Y_DIM] respectively
        x_, y_, log_w = self.sample_negatives(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x, y), with shape [...]
        unnormalized_log_ratio = self.unnormalized_log_ratio(x, y)

        x_dim = x_.ndim + (1 if isinstance(x_, torch.LongTensor) else 0)
        y_dim = y_.ndim + (1 if isinstance(y_, torch.LongTensor) else 0)

        assert x_dim == y_dim

        # Compute the log-normalization term log E{r(x,y)}[e^f(x,y)] on samples from r(x,y)
        log_partition = self.approx_log_partition(
            x=x, y=y, x_=x_, y_=y_, log_w=log_w
        )

        assert log_partition.shape == unnormalized_log_ratio.shape
        log_ratio = unnormalized_log_ratio - log_partition

        return log_ratio

    @abstractmethod
    def _approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_: torch.Tensor,
        log_w: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError()

    def approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_: torch.Tensor,
        y_: torch.Tensor,
        log_w: Optional[torch.Tensor],
    ) -> torch.Tensor:
        N = x.shape[0]
        M = self.n_negatives_to_use(N)

        # Evaluate the unnormalized_log_ratio f(x_,y_) on the samples x_, y_ ~ r(x, y). It has shape [M, ...]
        f_ = self.critic(x_, y_)

        # Computational shortcut for separable critic
        # we compute the product for the whole batch and then remove the diagonal
        if self.neg_samples < 0 and M != f_.shape[0]:
            assert isinstance(self.critic, SeparableCritic)
            # Remove the diagonal from the matrix
            f_ = matrix_off_diagonal(f_, M)

        assert f_.shape[0] == M and f_.shape[1] == N

        log_Z = self._approx_log_partition(x=x, y=y, f_=f_, log_w=log_w)
        assert log_Z.shape == x.shape[:-1]

        return log_Z

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
        y: torch.Tensor,
        f_: torch.Tensor,
        log_w: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Compute the baseline
        b = self.baseline(x=x, f_=f_).unsqueeze(0)
        assert (
            b.ndim == f_.ndim
        ), f"Baseline has ndim {b.ndim} while f_ has ndim {f_.ndim}"

        # Add the log_weights if provided
        if not (log_w is None):
            assert log_w.ndim == f_.ndim
            f_ = f_ + log_w

        # Compute the log_partition, it has shape [...]
        log_Z = (f_ - b).exp() + b - 1.0
        return log_Z.mean(0)

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
