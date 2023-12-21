from typing import List, Iterator, Optional, Dict, Any, Union

import numpy as np
import torch
from sklearn.cluster import KMeans

from torch_mist.distributions.factories import conditional_transformed_normal
from torch_mist.quantization.functions import LearnableVectorQuantization
from torch_mist.quantization.functions import ClusterQuantization
from torch_mist.quantization.vqvae import VQVAE


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
    input_dim: int,
    quantization_dim: int,
    n_bins: int,
    hidden_dims: List[int],
    beta: float = 0.2,
) -> VQVAE:
    quantization = vector_quantization(
        input_dim=input_dim,
        quantization_dim=quantization_dim,
        hidden_dims=hidden_dims,
        n_bins=n_bins,
    )

    decoder = conditional_transformed_normal(
        input_dim=input_dim,
        context_dim=quantization_dim,
        transform_name="conditional_linear",
        hidden_dims=hidden_dims,
    )

    return VQVAE(
        encoder=quantization,
        decoder=decoder,
        beta=beta,
    )


def vqvae_quantization(
    input_dim: int,
    n_bins: int,
    hidden_dims: List[int],
    data: Optional[Union[torch.Tensor, np.array]] = None,
    dataloader: Optional[Iterator] = None,
    quantization_dim: Optional[int] = None,
    beta: float = 0.2,
    max_epochs: int = 1,
    optimizer_class=torch.optim.Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 8,
) -> LearnableVectorQuantization:
    from torch_mist.utils.train.model import train_model

    if optimizer_params is None:
        optimizer_params = {"lr": 1e-3}

    model = vqvae(
        input_dim=input_dim,
        quantization_dim=quantization_dim,
        n_bins=n_bins,
        hidden_dims=hidden_dims,
        beta=beta,
    )

    train_model(
        model=model,
        dataloader=dataloader,
        data=data,
        max_epochs=max_epochs,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        batch_size=batch_size,
        num_workers=num_workers,
        logger=False,
    )

    return model.quantization
