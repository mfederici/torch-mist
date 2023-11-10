from typing import List, Iterator, Optional, Dict, Any, Union

import numpy as np
import torch
from sklearn.cluster import KMeans

from torch_mist.distributions.factories import conditional_transformed_normal
from torch_mist.quantization import LearnableVectorQuantization, VQVAE
from torch_mist.quantization.functions import ClusterQuantization


def vector_quantization(
    input_dim: int,
    n_bins: int,
    hidden_dims: List[int],
    quantization_dim: Optional[int] = None,
) -> LearnableVectorQuantization:
    assert len(hidden_dims) > 0, "hidden_dims must be a non-empty list"

    from pyro.nn import DenseNN

    if quantization_dim is None:
        quantization_dim = 16

    encoder = DenseNN(input_dim, hidden_dims, [quantization_dim])

    quantization = LearnableVectorQuantization(
        net=encoder, n_bins=n_bins, quantization_dim=quantization_dim
    )

    return quantization


def kmeans_quantization(
    data: Union[torch.Tensor, np.ndarray], n_bins: int, **kwargs
) -> ClusterQuantization:
    clustering = KMeans(n_bins, **kwargs)
    clustering.fit(data)

    return ClusterQuantization(clustering)


def vqvae(
    x_dim: int,
    quantization_dim: int,
    n_bins: int,
    hidden_dims: List[int],
    y_dim: Optional[int] = None,
    cross_modal: bool = False,
    decoder_transform_params: Optional[dict] = None,
    beta: float = 0.2,
) -> VQVAE:
    assert (
        not cross_modal or y_dim is not None
    ), "If cross_modal is True, y_dim must be provided"

    quantization = vector_quantization(
        input_dim=x_dim,
        quantization_dim=quantization_dim,
        hidden_dims=hidden_dims,
        n_bins=n_bins,
    )

    decoder = conditional_transformed_normal(
        input_dim=x_dim if not cross_modal else y_dim,
        context_dim=quantization_dim,
        transform_name="conditional_linear",
        transform_params=decoder_transform_params,
    )

    return VQVAE(
        encoder=quantization,
        decoder=decoder,
        cross_modal=cross_modal,
        beta=beta,
    )


def vqvae_quantization(
    input_dim: int,
    n_bins: int,
    hidden_dims: List[int],
    data: Optional[Union[torch.Tensor, np.array]] = None,
    dataloader: Optional[Iterator] = None,
    quantization_dim: Optional[int] = None,
    decoder_transform_params: Optional[Dict[str, Any]] = None,
    beta: float = 0.2,
    max_epochs: int = 1,
    optimizer_class=torch.optim.Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 8,
) -> LearnableVectorQuantization:
    from torch_mist.quantization.vqvae import VQVAE
    from torch_mist.train.vqvae import train_vqvae

    if optimizer_params is None:
        optimizer_params = {"lr": 1e-3}

    quantization = vector_quantization(
        input_dim=input_dim,
        n_bins=n_bins,
        hidden_dims=hidden_dims,
        quantization_dim=quantization_dim,
    )

    decoder = conditional_transformed_normal(
        input_dim=input_dim,
        context_dim=quantization_dim,
        transform_name="conditional_linear",
        transform_params=decoder_transform_params,
    )

    model = VQVAE(encoder=quantization, decoder=decoder, beta=beta)

    train_vqvae(
        model=model,
        dataloader=dataloader,
        data=data,
        max_epochs=max_epochs,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return quantization
