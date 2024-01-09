from typing import Union, List

import torch

from torch_mist.estimators import TransformedMIEstimator, MIEstimator


class FlippedMIEstimator(TransformedMIEstimator):
    def __init__(self, base_estimator: MIEstimator):
        super().__init__(base_estimator=base_estimator)

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.base_estimator.log_ratio(x=y, y=x)

    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.base_estimator.unnormalized_log_ratio(x=y, y=x)

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.base_estimator.batch_loss(x=y, y=x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.base_estimator.loss(x=y, y=x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.base_estimator.forward(x=y, y=x)
