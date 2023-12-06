from typing import Optional, Callable, Any, Dict, Union, List

import torch
from torch import nn

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.caching import (
    reset_cache_before_call,
    reset_cache_after_call,
    cached,
)
from torch_mist.utils.freeze import is_frozen


class DummyModule(nn.Module):
    def __init__(self, f: Callable[[Any], Any]):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __repr__(self):
        return str(self.f)


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
            if not base_estimator.infomax_gradient and (
                not is_frozen(transform)
            ):
                print(
                    "Warning: Transforms can be trained together with the estimator only when the estimator provides a valid infomax gradient."
                    f"The estimator {base_estimator.__class__} does not. You can use a different estimator or the transform form {variable}."
                )

            if not isinstance(transform, nn.Module):
                transform = DummyModule(transform)

            self.transforms[variable] = transform

    @cached
    def transform(self, **variables) -> Dict[str, torch.Tensor]:
        transformed_variables = {}
        for variable, transform in self.transforms.items():
            transformed_variables[variable] = transform(variables[variable])
        return transformed_variables

    def batch_loss(self, *args, **variables) -> torch.Tensor:
        if len(args) != 0 and len(args) != 2:
            raise ValueError(
                "The loss method can be called by passing two arguments or by specifying multiple named named arguments."
            )
        if len(args) == 2:
            if len(variables) > 0:
                raise ValueError(
                    "The loss method can be called by passing two arguments or by specifying multiple named named arguments."
                )
            variables["x"] = args[0]
            variables["y"] = args[1]

        variables.update(self.transform(**variables))
        return self.base_estimator.batch_loss(**variables)

    def log_ratio(
        self, *args, **variables
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if len(args) != 0 and len(args) != 2:
            raise ValueError(
                "The log_ratio method can be called by passing two arguments or by specifying multiple named named arguments."
            )
        if len(args) == 2:
            if len(variables) > 0:
                raise ValueError(
                    "The log_ratio method can be called by passing two arguments or by specifying multiple named named arguments."
                )
            variables["x"] = args[0]
            variables["y"] = args[1]

        variables.update(self.transform(**variables))
        return self.base_estimator.log_ratio(**variables)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.mutual_information(*args, **kwargs)
