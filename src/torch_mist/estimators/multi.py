from typing import List, Tuple, Dict

import torch
from torch import nn

from torch_mist.estimators.base import MIEstimator


class MultiMIEstimator(MIEstimator):
    def __init__(self, estimators: Dict[Tuple[str, str], MIEstimator]):
        super().__init__()

        self.estimators = nn.ModuleDict()
        for (x_key, y_key), estimator in estimators.items():
            key = f"{x_key};{y_key}"
            self.estimators[key] = estimator

    def broadcast_function(
        self, function_name: str, **variables
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        values = {}
        for key, estimator in self.estimators.items():
            x_key, y_key = key.split(";")
            if not (x_key in variables):
                raise ValueError(f"The variable {x_key} is not specified.")
            if not (y_key in variables):
                raise ValueError(f"The variable {y_key} is not specified.")

            x = variables[x_key]
            y = variables[y_key]

            value = getattr(estimator, function_name)(x=x, y=y)
            values[(x_key, y_key)] = value
        return values

    def loss(self, **variables) -> torch.Tensor:
        losses = self.broadcast_function("loss", **variables)
        total_loss: torch.Tensor = 0
        for loss in losses.values():
            total_loss += loss
        return total_loss

    def mutual_information(
        self, **variables
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        return self.broadcast_function("expected_log_ratio", **variables)

    def log_ratio(self, **variables) -> Dict[Tuple[str, str], torch.Tensor]:
        return self.broadcast_function("log_ratio", **variables)

    def forward(self, **variables) -> Dict[Tuple[str, str], torch.Tensor]:
        return self.mutual_information(**variables)
