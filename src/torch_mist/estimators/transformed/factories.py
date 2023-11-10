from typing import Optional, List

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.transformed.implementations import (
    PQ,
    BinnedMIEstimator,
)
from torch_mist.quantization import QuantizationFunction


def binned(
    Q_x: Optional[QuantizationFunction] = None,
    Q_y: Optional[QuantizationFunction] = None,
    temperature: float = 0.1,
) -> BinnedMIEstimator:
    return BinnedMIEstimator(
        Q_x=Q_x,
        Q_y=Q_y,
        temperature=temperature,
    )


def pq(
    Q_y: QuantizationFunction,
    x_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    q_QY_given_X: Optional[ConditionalDistribution] = None,
    temperature: float = 0.1,
) -> PQ:
    from torch_mist.distributions.factories import conditional_categorical

    if q_QY_given_X is None:
        if x_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_QY_given_X or x_dim and hidden_dims must be specified."
            )
        q_QY_given_X = conditional_categorical(
            n_classes=Q_y.n_bins,
            context_dim=x_dim,
            hidden_dims=hidden_dims,
            temperature=temperature,
        )

    return PQ(
        q_QY_given_X=q_QY_given_X,
        Q_y=Q_y,
        temperature=temperature,
    )
