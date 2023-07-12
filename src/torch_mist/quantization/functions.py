from typing import List, Iterator, Optional, Dict, Any

import torch
from torch import nn

from torch_mist.distributions.utils import conditional_transformed_normal


class QuantizationFunction(nn.Module):
    @property
    def n_bins(self) -> int:
        raise NotImplemented()

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        raise NotImplemented()


class LearnableQuantization(QuantizationFunction):
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()


class VectorQuantization(QuantizationFunction):
    def __init__(
            self,
            input_dim: int,
            n_bins: int,
    ):
        super().__init__()

        # Vectors used for quantization
        vectors = torch.zeros(n_bins, input_dim)
        vectors.uniform_(-1/n_bins, 1/n_bins)
        # self.vectors = nn.Parameter(vectors)
        self.register_buffer('vectors', vectors)

    def codebook_lookup(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.vectors.shape[-1]
        shape = x.shape[:-1]

        x = x.reshape(-1, 1, x.shape[-1])

        # Compute distances between z and vectors
        distances = torch.cdist(x, self.vectors.unsqueeze(0))
        # Get the closest vector
        _, indices = torch.min(distances, dim=-1)
        # Get the corresponding vector

        return indices.reshape(shape)

    @property
    def n_bins(self) -> int:
        return self.vectors.shape[0]

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        indices = self.codebook_lookup(x)
        return indices.long()


class LearnableVectorQuantization(VectorQuantization):
    def __init__(
            self,
            net: nn.Module,
            quantization_dim: int,
            n_bins: int,
    ):
        super().__init__(input_dim=quantization_dim, n_bins=n_bins)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        assert z.shape[-1] == self.vectors.shape[-1]
        return super().forward(z)

    def __repr__(self):
        s = f'{self.__class__.__name__}('
        s += f"\n  (net): "+self.net.__repr__().replace('\n', "\n  ")
        s += f"\n  (n_bins): {self.vectors.shape[0]}"
        s += "\n)"
        return s


class FixedQuantization(QuantizationFunction):
    def __init__(self, input_dim: int, thresholds: torch.Tensor):
        super().__init__()
        self.thresholds = thresholds
        self.input_dim = input_dim

    @property
    def n_bins(self) -> int:
        return (self.thresholds.shape[0]+1)**self.input_dim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        bins = torch.bucketize(x, self.thresholds)
        flat_bins = (bins * (torch.FloatTensor([self.n_bins]) ** torch.arange(self.input_dim)).to(bins.device)).sum(-1)
        return flat_bins.long()


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

    encoder = DenseNN(
        input_dim,
        hidden_dims,
        [quantization_dim]
    )

    quantization = LearnableVectorQuantization(
        net=encoder,
        n_bins=n_bins,
        quantization_dim=quantization_dim
    )

    return quantization


def vqvae_quantization(
        input_dim: int,
        n_bins: int,
        hidden_dims: List[int],
        dataloader: Iterator,
        quantization_dim: Optional[int] = None,
        cross_modal: bool = False,
        decoder_transform_params: Optional[Dict[str, Any]] = None,
        beta: float = 0.2,
        max_epochs: int = 1,
        optimizer_class=torch.optim.Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        target_dim: Optional[int] = None,
) -> LearnableVectorQuantization:

    from torch_mist.quantization.vqvae import VQVAE, train_vqvae

    assert not cross_modal or target_dim is not None, "target_dim must be specified if cross_modal is True"

    if optimizer_params is None:
        optimizer_params = {'lr': 1e-3}

    quantization = vector_quantization(
        input_dim=input_dim,
        n_bins=n_bins,
        hidden_dims=hidden_dims,
        quantization_dim=quantization_dim
    )

    decoder = conditional_transformed_normal(
        input_dim=input_dim if not cross_modal else target_dim,
        context_dim=quantization_dim,
        transform_name='conditional_linear',
        transform_params=decoder_transform_params
    )

    model = VQVAE(
        encoder=quantization,
        decoder=decoder,
        cross_modal=cross_modal,
        beta=beta
    )

    train_vqvae(
        model=model,
        dataloader=dataloader,
        max_epochs=max_epochs,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        cross_modal=cross_modal
    )

    return quantization
