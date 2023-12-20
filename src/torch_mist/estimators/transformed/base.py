from typing import Callable, Any, Dict, Union, List
from functools import lru_cache

import torch
from torch import nn

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.freeze import is_trainable
from torch_mist.estimators.transformed.utils import DummyModule

ERROR_MESSAGE = "The TransformedMIEstimator can be called by passing two arguments or by specifying multiple named named arguments."


class TransformedMIEstimator(MIEstimator):
    def __init__(
        self,
        base_estimator: MIEstimator,
        transforms: Dict[str, Callable[[Any], Any]],
    ):
        super().__init__()

        self.infomax_gradient = base_estimator.infomax_gradient

        self.base_estimator = base_estimator
        self.transforms = nn.ModuleDict()

        for variable, transform in transforms.items():
            if not base_estimator.infomax_gradient[variable] and (
                not is_trainable(transform)
            ):
                print(
                    "Warning: Transforms can be trained together with the estimator only when the estimator provides a valid infomax gradient."
                    f"The estimator {base_estimator.__class__} does not. You can use a different estimator or the transform for {variable}."
                )

            if not isinstance(transform, nn.Module):
                transform = DummyModule(transform)

            self.transforms[variable] = transform

    @lru_cache(maxsize=1)
    def transform(self, **variables) -> Dict[str, torch.Tensor]:
        transformed_variables = {}
        for variable, transform in self.transforms.items():
            transformed_variable = transform(variables[variable])

            # Detach the variables for which the gradient is not a valid infomax gradient
            if not self.infomax_gradient[variable]:
                transformed_variable = transformed_variable.detach()
            transformed_variables[variable] = transformed_variable

        return transformed_variables

    def _unfold_variables(self, *args, **variables) -> Dict[str, Any]:
        if len(args) != 0 and len(args) != 2:
            raise ValueError(ERROR_MESSAGE)
        if len(args) == 2:
            if len(variables) > 0:
                raise ValueError(ERROR_MESSAGE)
            variables["x"] = args[0]
            variables["y"] = args[1]

        return variables

    def batch_loss(self, *args, **variables) -> torch.Tensor:
        variables = self._unfold_variables(*args, **variables)
        variables.update(self.transform(**variables))
        return self.base_estimator.batch_loss(**variables)

    def log_ratio(
        self, *args, **variables
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        variables = self._unfold_variables(*args, **variables)
        variables.update(self.transform(**variables))
        return self.base_estimator.log_ratio(**variables)

    def unnormalized_log_ratio(self, *args, **variables) -> torch.Tensor:
        variables = self._unfold_variables(*args, **variables)
        variables.update(self.transform(**variables))
        return self.base_estimator.unnormalized_log_ratio(**variables)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.mutual_information(*args, **kwargs)
