from distutils.dist import Distribution
from typing import Dict, Sequence, Union, Callable, Any

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from torch_mist.distributions import JointDistribution


class SampleDataset(Sequence, Dataset):
    def __init__(self, samples: Dict[str, torch.Tensor]):
        self.samples = samples
        # Check all the samples have the same length
        lengths = [len(value) for value in samples.values()]
        assert all([length == lengths[0] for length in lengths])
        self.n_samples = lengths[0]

    def __getitem__(self, item):
        return {
            name: value[item]
            if value[item].ndim == 1
            else value[item].reshape(1)
            for name, value in self.samples.items()
        }

    def __len__(self):
        return self.n_samples


class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df is not an instance of pandas.DataFrame.")
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        values = {
            k: v
            if isinstance(v, np.ndarray)
            else np.array([v], dtype=np.int32).reshape(-1)
            if isinstance(v, int)
            else np.array([v], dtype=np.float32).reshape(-1)
            for k, v in dict(self.df.iloc[item]).items()
        }
        return values


class DistributionDataset(Dataset):
    def __init__(
        self,
        joint_dist: Union[Distribution, JointDistribution],
        max_samples: int = 100000,
        split_dim: int = -1,
    ):
        self.joint_dist = joint_dist
        self._n_samples = max_samples
        self.split_dim = split_dim

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(idx, int):
            sample_shape = torch.Size([])
        elif isinstance(idx, slice):
            sample_shape = torch.Size([len(idx.indices(self._n_samples))])
        else:
            raise ValueError("Please use int or slices for indexing.")

        samples = self.joint_dist.sample(sample_shape)
        return samples


class WrappedDataset(Dataset):
    def __init__(self, dataset: Dataset, func: Callable):
        super().__init__()
        self.dataset = dataset
        self.func = func

    def __getitem__(self, item) -> Any:
        return self.func(self.dataset.__getitem__(item))

    def __len__(self):
        return len(self.dataset)
