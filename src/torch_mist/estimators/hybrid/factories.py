from typing import Optional, List, Union, Dict, Any

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.factories import instantiate_estimator
from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.generative import GenerativeMIEstimator
from torch_mist.estimators.hybrid import (
    PQHybridMIEstimator,
    ResampledHybridMIEstimator,
    ReweighedHybridMIEstimator,
)
from torch_mist.quantization import (
    QuantizationFunction,
    instantiate_quantization,
)
from torch_mist.distributions.factories import conditional_categorical


def hybrid_pq(
    discriminative_estimator: Union[DiscriminativeMIEstimator, str],
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    quantize_y: Optional[Union[QuantizationFunction, str]] = None,
    hidden_dims: Optional[List[int]] = None,
    q_QY_given_X: Optional[ConditionalDistribution] = None,
    temperature: float = 0.1,
    n_bins: Optional[int] = 32,
    quantization_params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> PQHybridMIEstimator:
    if quantization_params is None:
        quantization_params = {}

    if isinstance(quantize_y, str):
        quantize_y = instantiate_quantization(
            quantize_y,
            n_bins=n_bins,
            **quantization_params,
        )

    if isinstance(discriminative_estimator, str):
        discriminative_estimator = instantiate_estimator(
            estimator_name=discriminative_estimator,
            hidden_dims=hidden_dims,
            x_dim=x_dim,
            y_dim=y_dim,
            **kwargs,
        )

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


def resampled_hybrid(
    discriminative_estimator: Union[DiscriminativeMIEstimator, str],
    generative_estimator: Union[GenerativeMIEstimator, str],
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    discriminative_params: Optional[Dict[str, Any]] = None,
    generative_params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ResampledHybridMIEstimator:
    if generative_params is None:
        generative_params = {}
    if discriminative_params is None:
        discriminative_params = {}

    if isinstance(generative_estimator, str):
        generative_estimator = instantiate_estimator(
            estimator_name=generative_estimator,
            hidden_dims=hidden_dims,
            x_dim=x_dim,
            y_dim=y_dim,
            **generative_params,
            **kwargs,
        )

    if not isinstance(generative_estimator, GenerativeMIEstimator):
        raise ValueError(
            f"Invalid generative_estimator: {generative_estimator}"
        )

    if isinstance(discriminative_estimator, str):
        discriminative_estimator = instantiate_estimator(
            estimator_name=discriminative_estimator,
            hidden_dims=hidden_dims,
            x_dim=x_dim,
            y_dim=y_dim,
            **discriminative_params,
            **kwargs,
        )

    return ResampledHybridMIEstimator(
        generative_estimator=generative_estimator,
        discriminative_estimator=discriminative_estimator,
    )


def reweighed_hybrid(
    discriminative_estimator: Union[DiscriminativeMIEstimator, str],
    generative_estimator: Union[GenerativeMIEstimator, str],
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    discriminative_params: Optional[Dict[str, Any]] = None,
    generative_params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> ReweighedHybridMIEstimator:
    if generative_params is None:
        generative_params = {}
    if discriminative_params is None:
        discriminative_params = {}

    if isinstance(generative_estimator, str):
        generative_estimator = instantiate_estimator(
            estimator_name=generative_estimator,
            hidden_dims=hidden_dims,
            x_dim=x_dim,
            y_dim=y_dim,
            **generative_params,
            **kwargs,
        )

    if not isinstance(generative_estimator, MIEstimator):
        raise ValueError(
            f"Invalid generative_estimator: {generative_estimator}"
        )

    if isinstance(discriminative_estimator, str):
        discriminative_estimator = instantiate_estimator(
            estimator_name=discriminative_estimator,
            hidden_dims=hidden_dims,
            x_dim=x_dim,
            y_dim=y_dim,
            **discriminative_params,
            **kwargs,
        )

    return ReweighedHybridMIEstimator(
        generative_estimator=generative_estimator,
        discriminative_estimator=discriminative_estimator,
    )
