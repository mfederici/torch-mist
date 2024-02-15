from typing import List, Optional, Callable

from sklearn.cluster import KMeans

from torch_mist.distributions.factories import conditional_transformed_normal
from torch_mist.nn import dense_nn
from torch_mist.quantization.functions import ClusterQuantization
from torch_mist.quantization.vqvae import VQVAE


def kmeans_quantization(
    n_bins: int, n_init="auto", **kwargs
) -> ClusterQuantization:
    clustering = KMeans(n_bins, n_init=n_init, **kwargs)
    return ClusterQuantization(clustering)


def vqvae(
    input_dim: int,
    quantization_dim: int,
    n_bins: int,
    hidden_dims: List[int],
    beta: float = 0.2,
    nonlinearity: Optional[Callable] = None,
    version: str = "v2",
    **train_params
) -> VQVAE:
    assert len(hidden_dims) > 0, "hidden_dims must be a non-empty list"

    if quantization_dim is None:
        quantization_dim = 16

    encoder = dense_nn(
        input_dim,
        output_dim=quantization_dim,
        hidden_dims=hidden_dims,
        nonlinearity=nonlinearity,
    )
    decoder = conditional_transformed_normal(
        input_dim=input_dim,
        context_dim=quantization_dim,
        transform_name="conditional_linear",
        hidden_dims=hidden_dims,
    )

    return VQVAE(
        encoder=encoder,
        decoder=decoder,
        n_bins=n_bins,
        quantization_dim=quantization_dim,
        beta=beta,
        version=version,
        **train_params
    )
