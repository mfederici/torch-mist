import inspect
from typing import Optional

from torch_mist.estimators.base import MIEstimator
from .generative.factories import *
from .discriminative.factories import *
from .transformed.factories import *
from .hybrid.factories import *

DEFAULT_HIDDEN_DIMS = [128]


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

    if "hidden_dims" in inspect.signature(
        factory_function
    ).parameters and not ("hidden_dims" in kwargs):
        kwargs["hidden_dims"] = DEFAULT_HIDDEN_DIMS
        print(
            f"[Info]: hidden_dims is not specified. Using hidden_dims={DEFAULT_HIDDEN_DIMS} by default."
        )

    estimator = factory_function(**kwargs)
    return estimator
