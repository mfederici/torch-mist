from contextlib import contextmanager

import torch

from torch_mist.baseline import BatchLogMeanExp, ExponentialMovingAverage
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)


class MINE(BaselineDiscriminativeMIEstimator):
    lower_bound = False  # Technically MINE is a lower bound, but sometimes it converges from above

    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
        gamma: float = 0.9,
    ):
        super().__init__(
            critic=critic,
            baseline=BatchLogMeanExp("all"),
            neg_samples=neg_samples,
        )
        self._train_baseline = ExponentialMovingAverage(gamma=gamma)

    @contextmanager
    def train_baseline(self):
        eval_baseline = self.baseline
        self.baseline = self._train_baseline
        try:
            yield
        finally:
            # Restore the original baseline
            self.baseline = eval_baseline

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with self.train_baseline():
            batch_loss = super().batch_loss(x, y)

        return batch_loss
