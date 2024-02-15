from typing import Tuple, Optional, Callable

import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid.base import HybridMIEstimator
from torch_mist.estimators.transformed.implementations.pq import PQ
from torch_mist.quantization import QuantizationFunction, LearnableQuantization
from torch_mist.utils.caching import cached_method


class PQHybridMIEstimator(HybridMIEstimator):
    def __init__(
        self,
        q_QY_given_X: ConditionalDistribution,
        quantize_y: QuantizationFunction,
        discriminative_estimator: DiscriminativeMIEstimator,
        temperature: float = 1.0,
    ):
        pq_estimator = PQ(
            q_QY_given_X=q_QY_given_X,
            quantize_y=quantize_y,
            temperature=temperature,
        )

        super().__init__(
            generative_estimator=pq_estimator,
            discriminative_estimator=discriminative_estimator,
        )
        self._validate_batches = True

    def disable_batch_validation(self):
        self._validate_batches = False

    def enable_batch_validation(self):
        self._validate_batches = True

    @property
    def quantize_y(self) -> Callable[[torch.Tensor], torch.LongTensor]:
        return self.generative_estimator.transforms["y->y"]

    @cached_method
    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # We reshape the batches from [batch_size, ...] to [neg_samples+1, batch_size//neg_samples, ...]
        # This is because the batches are created so that the negatives can be sampled by shuffling the new first dim
        if self.neg_samples <= 0:
            neg_shape = x.shape[0]
        else:
            neg_shape = self.neg_samples + 1

        batch_shape = x.shape[0] // (neg_shape)

        # If the batch validation is disabled, we skip checking the batches to speed up computation
        if self._validate_batches:
            Q_y = self.quantize_y(y)
            Q_y = Q_y.view(neg_shape, batch_shape, *Q_y.shape[1:])

            # Check the labels are the same for the corresponding ys, (this can be done using the SameAttributeSampler)
            Q_y0 = Q_y[0]
            assert (
                torch.sum(Q_y0.unsqueeze(0) == Q_y).item() == Q_y.numel()
            ), Q_y

        # Sample negatives as usual by shuffling the (new) first dimension
        # this way we shuffle only y with the same Q_y
        x_, y_, log_w = DiscriminativeMIEstimator.sample_negatives(
            self,
            x.view(neg_shape, batch_shape, *x.shape[1:]),
            y.view(neg_shape, batch_shape, *y.shape[1:]),
        )

        x_ = x.unsqueeze(0)
        # Collapse the intermediate dimensions to obtain [neg_samples, batch_size, ...]
        y_ = y_.reshape(y_.shape[0], -1, *y_.shape[3:])
        if log_w:
            log_w = log_w.reshape(log_w.shape[0], -1, *log_w.shape[3:])

        return x_, y_, log_w
