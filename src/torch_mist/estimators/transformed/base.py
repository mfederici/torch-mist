from typing import Callable, Any, Dict, Union, List, Tuple, Optional

import torch
from torch import nn

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.caching import cached_method
from torch_mist.utils.freeze import is_trainable
from torch_mist.estimators.transformed.utils import DummyModule

ERROR_MESSAGE = "The TransformedMIEstimator can be called by passing two arguments or by specifying multiple named named arguments."
SPLIT_SEQUENCE = "->"


class TransformedMIEstimator(MIEstimator):
    def __init__(
        self,
        base_estimator: MIEstimator,
        transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
        transforms_rename: Optional[
            Dict[Tuple[str, str], Callable[[Any], Any]]
        ] = None,
    ):
        super().__init__()

        assert transforms or transforms_rename
        if transforms_rename is None:
            transforms_rename = {}

        if transforms is None:
            transforms = {}

        self.infomax_gradient = base_estimator.infomax_gradient
        self.upper_bound = False
        self.lower_bound = base_estimator.lower_bound

        self.base_estimator = base_estimator
        self.transforms = nn.ModuleDict()

        additional_transforms = {
            (name, name): transform for name, transform in transforms.items()
        }
        transforms_rename.update(additional_transforms)
        for (
            variable_from,
            variable_to,
        ), transform in transforms_rename.items():
            if not base_estimator.infomax_gradient[variable_to] and (
                is_trainable(transform)
            ):
                print(
                    "Warning: Transforms can be trained together with the estimator only when the estimator provides a valid infomax gradient."
                    f"The estimator {base_estimator.__class__} does not. You can use a different estimator or the transform for {variable_from}."
                )

            if not isinstance(transform, nn.Module):
                transform = DummyModule(transform)

            if (
                SPLIT_SEQUENCE in variable_to
                or SPLIT_SEQUENCE in variable_from
            ):
                raise ValueError(
                    f"The estimators does not support variables that contain '{SPLIT_SEQUENCE}'."
                )

            self.transforms[
                f"{variable_from}{SPLIT_SEQUENCE}{variable_to}"
            ] = transform
            self.infomax_gradient[
                variable_from
            ] = base_estimator.infomax_gradient[variable_to]

    @cached_method
    def transform(self, **variables) -> Dict[str, torch.Tensor]:
        transformed_variables = {}
        for variable_fromto, transform in self.transforms.items():
            variable_from, variable_to = variable_fromto.split(SPLIT_SEQUENCE)
            transformed_variable = transform(variables[variable_from])

            # Detach the variables for which the gradient is not a valid infomax gradient
            if not self.infomax_gradient[variable_to]:
                transformed_variable = transformed_variable.detach()
            transformed_variables[variable_to] = transformed_variable

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

    def loss(self, *args, **variables) -> torch.Tensor:
        return self.batch_loss(*args, **variables).mean()

    def mutual_information(
        self, *args, **variables
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        log_ratio = self.log_ratio(**variables)
        if isinstance(log_ratio, dict):
            return {k: v.mean() for k, v in log_ratio.items()}
        else:
            return log_ratio.mean()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.loss(*args, **kwargs)
