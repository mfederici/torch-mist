from abc import abstractmethod
from functools import lru_cache

import torch

from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.utils.freeze import is_trainable


class HybridMIEstimator(DiscriminativeMIEstimator):
    def __init__(
        self,
        generative_estimator: MIEstimator,
        discriminative_estimator: DiscriminativeMIEstimator,
    ):
        super().__init__(
            critic=discriminative_estimator.critic,
            neg_samples=discriminative_estimator.neg_samples,
        )

        self.discriminative_estimator = discriminative_estimator
        self.generative_estimator = generative_estimator

        informax_gradient = generative_estimator.infomax_gradient
        informax_gradient = {
            key: value and discriminative_estimator.infomax_gradient[key]
            for key, value in informax_gradient.items()
        }
        self.infomax_gradient = informax_gradient

    @lru_cache(maxsize=1)
    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        f = self.critic(x, y)
        assert f.ndim == y.ndim - 1

        partial_log_ratio = self.generative_estimator.log_ratio(x, y)

        assert f.shape == partial_log_ratio.shape
        return f + partial_log_ratio

    def _approx_log_partition(
        self, x: torch.Tensor, f_: torch.Tensor
    ) -> torch.Tensor:
        return self.discriminative_estimator._approx_log_partition(x, f_)

    @abstractmethod
    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute only the discriminative loss component
        # Note that we can skip computing the log r(y|x)/p(y)
        unnormalized_log_ratio = (
            self.discriminative_estimator.unnormalized_log_ratio(x, y)
        )
        log_partition = self.approx_log_partition(x, y)

        # The loss is the same as for generative estimators with the difference in the computation for the normalization
        # constant
        batch_loss = -(unnormalized_log_ratio - log_partition)

        # If the generative component is not trainable, there is no need to compute the log-ratio or the generative loss
        if not is_trainable(self.generative_estimator):
            batch_loss += self.generative_estimator.batch_loss(x, y)

        return batch_loss


class ReWeighedHybridMIEstimator(HybridMIEstimator):
    @lru_cache(maxsize=1)
    def approx_log_partition(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        y_ = self.sample_negatives(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)r(y|x)
        # The tensor f_ has shape [M, N...] in which f_[i,j] contains critic(x[j], y_[i,j]).
        # and y_ is sampled from r(y|x), which is set to the empirical p(y) unless a proposal is specified
        f_ = self.critic(x, y_)

        log_Z = self.discriminative_estimator._approx_log_partition(x, f_)

        weights = self.generative_estimator.log_ratio(x.unsqueeze(0), y_).exp()
        assert log_Z.shape == weights.shape
        log_Z = log_Z * weights.detach()

        assert log_Z.shape[0] == self.n_negatives_to_use(x.shape[0])
        assert (
            not isinstance(x, torch.LongTensor)
            and log_Z.shape[1:] == x.shape[:-1]
        ) or (isinstance(x, torch.LongTensor) and log_Z.shape[1:] == x.shape)

        return log_Z.mean(0)
