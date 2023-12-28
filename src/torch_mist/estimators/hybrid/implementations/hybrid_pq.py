from typing import Tuple, Optional

import torch

from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid.base import HybridMIEstimator
from torch_mist.estimators.transformed.implementations.pq import PQ


class PQHybridMIEstimator(HybridMIEstimator):
    def __init__(
        self,
        pq_estimator: PQ,
        discriminative_estimator: DiscriminativeMIEstimator,
    ):
        super().__init__(
            generative_estimator=pq_estimator,
            discriminative_estimator=discriminative_estimator,
        )
        self._validate_batches = True

    def disable_batch_validation(self):
        self._validate_batches = False

    def enable_batch_validation(self):
        self._validate_batches = True

    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # We reshape the batches from [batch_size, ...] to [neg_samples+1, batch_size//neg_samples, ...]
        # This is because the batches are created so that the negatives can be sampled by shuffling the new first dim
        neg_shape = self.neg_samples + 1
        batch_shape = x.shape[0] // (self.neg_samples + 1)

        # If the batch validation is disabled, we skip checking the batches to speed up computation
        if self._validate_batches:
            discretize = self.generative_estimator.transforms["y->y"]
            Q_y = discretize(y)
            Q_y = Q_y.view(neg_shape, batch_shape, *Q_y.shape[1:])

            # Check the labels are the same for the corresponding ys, (this can be done using the SameAttributeSampler)
            Q_y0 = Q_y[0]
            assert (
                torch.sum(Q_y0.unsqueeze(0) == Q_y).item() == Q_y.numel()
            ), Q_y

        # Sample negatives as usual by shuffling the (new) first dimension
        # this way we shuffle only y with the same Q_y
        y_, w = DiscriminativeMIEstimator.sample_negatives(
            self,
            x.view(neg_shape, batch_shape, *x.shape[1:]),
            y.view(neg_shape, batch_shape, *y.shape[1:]),
        )

        # Collapse the intermediate dimensions to obtain [neg_samples, batch_size, ...]
        y_ = y_.reshape(y_.shape[0], -1, *y_.shape[3:])
        if w:
            w = w.reshape(w.shape[0], -1, *w.shape[3:])

        return y_, w
