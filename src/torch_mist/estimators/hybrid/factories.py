from typing import Optional, List

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid import (
    PQHybridMIEstimator,
)
from torch_mist.quantization import QuantizationFunction
from torch_mist.distributions.factories import conditional_categorical


def hybrid_pq(
    discriminative_estimator: DiscriminativeMIEstimator,
    x_dim: int,
    quantize_y: Optional[QuantizationFunction] = None,
    hidden_dims: Optional[List[int]] = None,
    q_QY_given_X: Optional[ConditionalDistribution] = None,
    temperature: float = 0.1,
) -> PQHybridMIEstimator:
    if q_QY_given_X is None:
        if x_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_QY_given_X or x_dim and hidden_dims must be specified."
            )
        q_QY_given_X = conditional_categorical(
            n_classes=quantize_y.n_bins,
            context_dim=x_dim,
            hidden_dims=hidden_dims,
            temperature=temperature,
        )

    return PQHybridMIEstimator(
        quantize_y=quantize_y,
        q_QY_given_X=q_QY_given_X,
        discriminative_estimator=discriminative_estimator,
        temperature=temperature,
    )
