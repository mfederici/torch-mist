from typing import Tuple
from collections.abc import Iterator

import torch
from torch.distributions import Distribution


class SampleDataLoader(Iterator[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
            self,
            joint_dist: Distribution,
            batch_size: int,
            max_samples: int = 100000,
            split_dim: int = -1,
    ):
        self.joint_dist = joint_dist
        self.split_dim = split_dim
        self.batch_size = batch_size
        self.max_samples = max_samples
        self._n_samples = max_samples

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._n_samples <= 0:
            self._n_samples = self.max_samples
            raise StopIteration
        else:
            self._n_samples -= self.batch_size
            x, y = torch.chunk(
                self.joint_dist.sample(torch.Size([self.batch_size])),
                2,
                self.split_dim
            )
            x = x.squeeze(self.split_dim)
            y = y.squeeze(self.split_dim)

            return x, y

    def __len__(self):
        return self.max_samples // self.batch_size



