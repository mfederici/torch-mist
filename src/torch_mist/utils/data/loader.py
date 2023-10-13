from typing import Dict, Union, Sequence
from collections import Iterator

import numpy as np
import torch
from torch.distributions import Distribution
from torch.utils.data import DataLoader, Dataset

from torch_mist.distributions.joint import JointDistribution
from torch_mist.utils.data.sampler import SameAttributeSampler


class DistributionDataLoader(Iterator[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        joint_dist: Union[Distribution, JointDistribution],
        batch_size: int,
        max_samples: int = 100000,
        split_dim: int = -1,
    ):
        self.joint_dist = joint_dist
        self.split_dim = split_dim
        self.batch_size = batch_size
        self.max_samples = max_samples
        self._n_samples = max_samples

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self._n_samples <= 0:
            self._n_samples = self.max_samples
            raise StopIteration
        else:
            n_samples = min(self.batch_size, self._n_samples)

            samples = self.joint_dist.sample(torch.Size([n_samples]))

            if isinstance(samples, tuple):
                assert len(samples) == 2
                samples = {"x": samples[0], "y": samples[1]}

            self._n_samples -= n_samples

            assert isinstance(samples, dict)

            return samples

    def __iter__(self):
        return DistributionDataLoader(
            joint_dist=self.joint_dist,
            batch_size=self.batch_size,
            max_samples=self.max_samples,
            split_dim=self.split_dim,
        )

    def __len__(self):
        return self.max_samples // self.batch_size


class SameAttributeDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        attributes: Union[torch.Tensor, np.ndarray],
        batch_size: int,
        **kwargs
    ):
        if len(dataset) != len(attributes):
            raise Exception(
                "The dataset and the attributes must have the same length."
            )

        super().__init__(
            dataset,
            batch_sampler=SameAttributeSampler(
                attributes=attributes, batch_size=batch_size
            ),
            **kwargs
        )


def sample_same_attributes(
    dataloader: DataLoader,
    attributes: Union[torch.Tensor, np.ndarray],
):
    return SameAttributeDataLoader(
        dataset=dataloader.dataset,
        attributes=attributes,
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        prefetch_factor=dataloader.prefetch_factor,
        pin_memory=dataloader.pin_memory,
        pin_memory_device=dataloader.pin_memory_device,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
    )