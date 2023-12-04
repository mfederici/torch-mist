import torch

from torch_mist.estimators import MIEstimator


class FlippedMIEstimator(MIEstimator):
    def __init__(self, estimator: MIEstimator):
        super().__init__()
        self.estimator = estimator

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.estimator.log_ratio(y, x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.estimator.loss(y, x)

    def mutual_information(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.estimator.mutual_information(y, x)


def flip_estimator(estimator: MIEstimator) -> MIEstimator:
    if isinstance(estimator, FlippedMIEstimator):
        return estimator.estimator
    else:
        return FlippedMIEstimator(estimator)
