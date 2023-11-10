from torch_mist.estimators.base import MIEstimator
from .generative.factories import *
from .discriminative.factories import *
from .transformed.factories import *


def instantiate_estimator(estimator_name, **kwargs) -> MIEstimator:
    factory_function = globals().get(estimator_name)

    # Check if the function name exists in the global namespace
    if not factory_function:
        raise ValueError(f"Factory function {estimator_name} not found.")
    estimator = factory_function(**kwargs)
    return estimator
