from abc import abstractmethod
from typing import Optional, Any

import numpy as np
from sklearn.base import TransformerMixin
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_mist.utils.freeze import freeze


class QuantizationFunction(nn.Module):
    @property
    def n_bins(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> torch.LongTensor:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        return self.quantize(x)


class NotTrainedError(Exception):
    def __init__(self, message: str, model_to_train: nn.Module):
        super().__init__(message)
        self.model_to_train = model_to_train


class LearnableQuantization(QuantizationFunction):
    def __init__(self, **train_params):
        super().__init__()
        self.trained = False
        self.train_params = train_params

    def _fit(self, dataloader: DataLoader) -> Optional[Any]:
        from torch_mist.utils.train.model import train_model

        return train_model(
            model=self, train_data=dataloader, **self.train_params
        )

    def fit(self, dataloader: Any) -> Optional[Any]:
        log = self._fit(dataloader)
        self.trained = True
        freeze(self)
        return log

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        if not self.trained:
            raise NotTrainedError(
                f"The {self.__class__.__name__} quantization scheme needs to be trained.\n"
                + "Please call the fit(data, **kwargs) method before quantizing.",
                self,
            )
        return super().forward(x)

    def loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class ClusterQuantization(LearnableQuantization):
    def __init__(self, clustering: TransformerMixin, **train_params):
        super().__init__(**train_params)
        self.clustering = clustering

    @property
    def n_bins(self) -> int:
        return self.clustering.n_clusters

    def _fit(self, dataloader: DataLoader):
        data = []
        for batch in dataloader:
            assert isinstance(batch, torch.Tensor)
            data.append(batch.cpu().data.numpy())

        data = np.concatenate(data, 0).astype(np.float32)

        if not isinstance(data, np.ndarray):
            raise ValueError(
                "data needs to be an instance of torch.Tensor or np.array."
            )

        self.clustering.fit(data, **self.train_params)

    def quantize(self, x: torch.Tensor) -> torch.LongTensor:
        device = x.device
        shape = x.shape[:-1]
        feature_dim = x.shape[-1]
        x = x.view(-1, feature_dim)

        if isinstance(x, torch.Tensor):
            x = x.cpu().data.numpy()

        return (
            torch.LongTensor(self.clustering.predict(x)).view(shape).to(device)
        )


class VectorQuantization(QuantizationFunction):
    def __init__(
        self,
        vectors: torch.Tensor,
    ):
        super().__init__()
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

    def quantize(self, x: torch.Tensor) -> torch.LongTensor:
        indices = self.codebook_lookup(x)
        return indices.long()


class LearnableVectorQuantization(VectorQuantization, LearnableQuantization):
    def __init__(
        self,
        transform: Optional[nn.Module] = None,
        vectors: Optional[torch.Tensor] = None,
        n_bins: Optional[int] = None,
        quantization_dim: Optional[int] = None,
        **train_params,
    ):
        if vectors is None:
            if n_bins is None or quantization_dim is None:
                raise ValueError(
                    "Either `vectors` or `n_bins` and `input_dim` need to be specified"
                )

            # Vectors used for quantization
            vectors = torch.zeros(n_bins, quantization_dim)
            vectors.uniform_(-1.0 / n_bins, 1.0 / n_bins)
        else:
            if vectors.ndim != 2:
                raise NotImplementedError(
                    "Vector quantization is currently supported only on one dimension"
                )
            if quantization_dim:
                if quantization_dim != vectors.shape[-1]:
                    raise ValueError(
                        "The specified quantization_dim is not compatible with the vectors."
                    )
            if n_bins:
                if n_bins != vectors.shape[0]:
                    raise ValueError(
                        "The specified n_bins differs from the number of quantization vectors"
                    )

        VectorQuantization.__init__(self, vectors=vectors)
        self.train_params = train_params
        self.trained = False
        self.transform = transform

    def quantize(self, x: torch.Tensor) -> torch.LongTensor:
        if self.transform:
            x = self.transform(x)
        return super().quantize(x)


class FixedQuantization(QuantizationFunction):
    def __init__(self, input_dim: int, thresholds: torch.Tensor):
        super().__init__()
        self.thresholds = thresholds
        self.input_dim = input_dim

    @property
    def n_bins(self) -> int:
        return (self.thresholds.shape[0] + 1) ** self.input_dim

    def quantize(self, x: torch.Tensor) -> torch.LongTensor:
        bins = torch.bucketize(x, self.thresholds)
        flat_bins = (
            bins
            * (
                torch.FloatTensor([self.thresholds.shape[0] + 1])
                ** torch.arange(self.input_dim)
            ).to(bins.device)
        ).sum(-1)
        return flat_bins.long()
