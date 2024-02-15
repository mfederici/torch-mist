from typing import List, Tuple, Dict, Callable

import torch
from torch import nn

from torch_mist.estimators.base import MIEstimator


class MultiMIEstimator(MIEstimator):
    def __init__(self, estimators: Dict[Tuple[str, str], MIEstimator]):
        super().__init__()

        self.estimators = nn.ModuleDict()
        self.infomax_gradient = {}
        self.lower_bound = True
        self.upper_bound = True
        for (x_key, y_key), estimator in estimators.items():
            self.upper_bound = self.upper_bound and estimator.upper_bound
            self.lower_bound = self.lower_bound and estimator.lower_bound

            # Add the renaming for the components that need pre-training (if needed)
            for data_process, component in estimator._components_to_pretrain:

                def new_data_process(batch: Dict[str, torch.Tensor]):
                    batch["x"] = batch[x_key]
                    batch["y"] = batch[y_key]
                    return data_process(batch)

                self._components_to_pretrain.append(
                    (new_data_process, component)
                )

            key = f"{x_key};{y_key}"
            self.estimators[key] = estimator
            if x_key in self.infomax_gradient:
                self.infomax_gradient[x_key] = (
                    self.infomax_gradient[x_key]
                    and estimator.infomax_gradient["x"]
                )
            else:
                self.infomax_gradient[x_key] = estimator.infomax_gradient["x"]
            if y_key in self.infomax_gradient:
                self.infomax_gradient[y_key] = (
                    self.infomax_gradient[y_key]
                    and estimator.infomax_gradient["y"]
                )
            else:
                self.infomax_gradient[y_key] = estimator.infomax_gradient["y"]

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

            value = getattr(estimator, function_name)(x, y)
            values[(x_key, y_key)] = value
        return values

    def loss(self, **variables) -> torch.Tensor:
        losses = self.broadcast_function("loss", **variables)
        total_loss: torch.Tensor = 0
        for loss in losses.values():
            total_loss += loss
        return total_loss

    def batch_loss(self, **variables) -> torch.Tensor:
        losses = self.broadcast_function("batch_loss", **variables)
        total_loss: torch.Tensor = 0
        for loss in losses.values():
            total_loss += loss
        return total_loss

    def mutual_information(
        self, **variables
    ) -> Dict[Tuple[str, str], torch.Tensor]:
        return self.broadcast_function("mutual_information", **variables)

    def log_ratio(self, **variables) -> Dict[Tuple[str, str], torch.Tensor]:
        return self.broadcast_function("log_ratio", **variables)

    def forward(self, **variables) -> torch.Tensor:
        return self.loss(**variables)
