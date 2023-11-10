from typing import Optional

from sklearn.base import TransformerMixin
import torch
from torch import nn


class QuantizationFunction(nn.Module):
    @property
    def n_bins(self) -> int:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        raise NotImplementedError()


class LearnableQuantization(QuantizationFunction):
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class ClusterQuantization(QuantizationFunction):
    def __init__(self, clustering: TransformerMixin):
        super().__init__()
        self.clustering = clustering

    @property
    def n_bins(self) -> int:
        return self.clustering.n_clusters

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        device = x.device
        if isinstance(x, torch.Tensor):
            x = x.cpu()

        shape = x.shape[:-1]
        feature_dim = x.shape[-1]

        return (
            torch.LongTensor(self.clustering.predict(x.view(-1, feature_dim)))
            .view(shape)
            .to(device)
        )


class VectorQuantization(QuantizationFunction):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        n_bins: Optional[int] = None,
        vectors: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if vectors is None:
            if n_bins is None or input_dim is None:
                raise ValueError(
                    "Either `vectors` or `n_bins` and `input_dim` need to be specified"
                )

            # Vectors used for quantization
            vectors = torch.zeros(n_bins, input_dim)
            vectors.uniform_(-1 / n_bins, 1 / n_bins)
            # self.vectors = nn.Parameter(vectors)
        else:
            if not (n_bins is None or input_dim is None):
                raise ValueError(
                    "Either `vectors` or `n_bins` and `input_dim` need to be specified"
                )
        self.register_buffer("vectors", vectors)

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
        s = f"{self.__class__.__name__}("
        s += f"\n  (net): " + self.net.__repr__().replace("\n", "\n  ")
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
        return (self.thresholds.shape[0] + 1) ** self.input_dim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        bins = torch.bucketize(x, self.thresholds)
        flat_bins = (
            bins
            * (
                torch.FloatTensor([self.thresholds.shape[0] + 1])
                ** torch.arange(self.input_dim)
            ).to(bins.device)
        ).sum(-1)
        return flat_bins.long()
