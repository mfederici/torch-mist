import inspect
from typing import Optional

from torch_mist.estimators.base import MIEstimator
from .generative.factories import *
from .discriminative.factories import *
from .transformed.factories import *
from .hybrid.factories import *


def instantiate_estimator(
    estimator_name: str,
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    **kwargs,
) -> MIEstimator:
    factory_function = globals().get(estimator_name)

    # Check if the function name exists in the global namespace
    if not factory_function:
        raise ValueError(f"Factory function {estimator_name} not found.")

    if "x_dim" in inspect.signature(factory_function).parameters:
        kwargs["x_dim"] = x_dim
    if "y_dim" in inspect.signature(factory_function).parameters:
        kwargs["y_dim"] = y_dim

    estimator = factory_function(**kwargs)
    return estimator
