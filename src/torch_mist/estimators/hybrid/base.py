from abc import abstractmethod
from functools import lru_cache
from typing import Optional, Tuple

import torch

from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.utils.freeze import is_trainable
from contextlib import contextmanager


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
        partial_log_ratio = self.generative_estimator.log_ratio(x, y)
        f = self.discriminative_estimator.unnormalized_log_ratio(x, y)

        assert f.shape == partial_log_ratio.shape
        return f + partial_log_ratio

    @lru_cache(maxsize=1)
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        partial_log_ratio = self.generative_estimator.log_ratio(x, y)

        with self.resampling_strategy():
            log_rest = self.discriminative_estimator.log_ratio(x, y)

        assert partial_log_ratio.shape == log_rest.shape
        return partial_log_ratio + log_rest

    @abstractmethod
    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()

    @contextmanager
    def resampling_strategy(self):
        # Store the original sampling scheme
        original_method = self.discriminative_estimator.sample_negatives

        # Replace it with the new one
        self.discriminative_estimator.sample_negatives = self.sample_negatives
        try:
            yield
        finally:
            # Restore the original method
            self.discriminative_estimator.sample_negatives = original_method

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute only the discriminative loss component
        # Note that we can skip computing the log r(y|x)/p(y)
        with self.resampling_strategy():
            batch_loss = self.discriminative_estimator.batch_loss(x, y)

        # If the generative component is not trainable, there is no need to compute the log-ratio or the generative loss
        if is_trainable(self.generative_estimator):
            batch_loss += self.generative_estimator.batch_loss(x, y)

        return batch_loss

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  (generative_estimator): "
            + str(self.generative_estimator).replace("\n", "\n" + "  ")
            + "\n"
        )
        s += (
            "  (discriminative_estimator): "
            + str(self.discriminative_estimator).replace("\n", "\n" + "  ")
            + "\n"
        )
        s += ")"

        return s
