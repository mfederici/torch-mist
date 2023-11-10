from typing import Optional, Callable, Any

import torch

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.caching import (
    reset_cache_before_call,
    reset_cache_after_call,
    cached,
)
from torch_mist.utils.freeze import is_frozen


class TransformedMIEstimator(MIEstimator):
    def __init__(
        self,
        base_estimator: MIEstimator,
        f_x: Optional[Callable[[Any], Any]] = None,
        f_y: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__()

        if not base_estimator.infomax_gradient and (
            not is_frozen(f_x) or not is_frozen(f_y)
        ):
            raise ValueError(
                "f_x and f_y can be trained together with the estimator only when the estimator provides a valid infomax gradient."
                f"The estimator {base_estimator.__class__} does not. You can use a different estimator or freeze f_x and f_y."
            )

        self.infomax_gradient = base_estimator.infomax_gradient

        self.base_estimator = base_estimator
        self.f_x = f_x
        self.f_y = f_y

    @cached
    def transform(self, x: Any, y: Any):
        if self.f_x:
            x = self.f_x(x)
        if self.f_y:
            y = self.f_y(y)
        return x, y

    @reset_cache_before_call
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = self.transform(x, y)
        return self.base_estimator.loss(x, y)

    def expected_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        x, y = self.transform(x, y)
        return self.base_estimator.expected_log_ratio(x, y)

    @reset_cache_after_call
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = self.transform(x, y)
        return self.base_estimator.log_ratio(x, y)
