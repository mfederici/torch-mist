from typing import Tuple, List, Iterator, Optional, Dict, Any

import torch
from pyro.distributions import ConditionalDistribution
from torch import nn
from torch.distributions import RelaxedOneHotCategorical


class QuantizationFunction(nn.Module):
    @property
    def n_bins(self) -> int:
        raise NotImplemented()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = self.codebook_lookup(x)
        return torch.nn.functional.one_hot(indices.to(torch.int64), self.n_bins).float()


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bins = torch.bucketize(x, self.thresholds)
        flat_bins = (bins * (torch.FloatTensor([self.n_bins]) ** torch.arange(self.input_dim)).to(bins.device)).sum(-1)
        return torch.nn.functional.one_hot(flat_bins.to(torch.int64), self.n_bins).float()


def learnable_vector_quantization(
    input_dim: int,
    quantization_dim: int,
    hidden_dims: List[int],
    n_bins: int,
) -> LearnableVectorQuantization:
    from pyro.nn import DenseNN

    net = DenseNN(
        input_dim,
        hidden_dims,
        [quantization_dim]
    )

    return LearnableVectorQuantization(
        net=net,
        n_bins=n_bins,
        quantization_dim=quantization_dim
    )


def trained_vector_quantization(
        dataloader: Iterator,
        x_dim: int,
        n_bins: int,
        hidden_dims: List[int],
        quantization_dim: Optional[int] = None,
        cross_modal: bool = False,
        decoder_transform_params: Optional[Dict[str, Any]] = None,
        beta: float = 0.2,
        n_train_epochs: int = 1,
        optimizer_class=torch.optim.Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        y_dim: Optional[int] = None,
) -> QuantizationFunction:

    assert not cross_modal or y_dim is not None, "y_dim must be specified if cross_modal is True"
    assert len(hidden_dims) > 0, "hidden_dims must be a non-empty list"

    if optimizer_params is None:
        optimizer_params = {'lr': 1e-3}

    from torch_mist.quantization import vqvae
    from tqdm.auto import tqdm

    if quantization_dim is None:
        quantization_dim = 16

    model = vqvae(
        x_dim=x_dim,
        y_dim=y_dim,
        quantization_dim=quantization_dim,
        n_bins=n_bins,
        hidden_dims=hidden_dims,
        decoder_transform_params=decoder_transform_params,
        cross_modal=cross_modal,
        beta=beta
    )

    opt = optimizer_class(model.parameters(), **optimizer_params)

    for epoch in range(n_train_epochs):
        for data in tqdm(dataloader):
            opt.zero_grad()
            model.loss(*data).backward()
            opt.step()

    return model.quantization




