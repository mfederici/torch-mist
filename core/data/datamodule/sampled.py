from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.data.datamodule.base import DataModuleWithAttributes
from core.data.datasets import SampleDataset
from core.data.datasets.attributes import AttributeDataset
from core.data.sampler import SameAttributesSampler
from core.distributions.test.multivariate_normal_mixture import JointDistribution





class SampledDataModule(DataModuleWithAttributes):
    def __init__(
            self,
            dist: JointDistribution,
            batch_size: int = 64,
            num_workers: int = 1,
            n_samples_train: int = 100000,
            n_samples_val: int = 10000,
            n_samples_test: int = 500000,
            compute_attributes: Optional[Callable[[torch.Tensor], torch.LongTensor]] = None,
    ):
        super().__init__()

        self.dist = dist
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.n_samples_train = n_samples_train
        self.n_samples_val = n_samples_val
        self.n_samples_test = n_samples_test
        self.compute_attributes = compute_attributes

    def sample_dataset(self, n_samples: int) -> AttributeDataset:
        # Sample a train set
        dataset = SampleDataset(
            distribution=self.dist,
            n_samples=n_samples,
        )

        # Compute the corresponding attributes
        if self.compute_attributes is not None:
            attributes = self.compute_attributes(dataset.samples['x'])
        else:
            attributes = None

        return AttributeDataset(
            dataset=dataset,
            attributes=attributes
        )

    def prepare_data(self) -> None:
        self.train_set = self.sample_dataset(self.n_samples_train)
        self.train_sampler = SameAttributesSampler(
            attributes=self.train_set.attributes,
            n_samples=self.n_samples_train,
            batch_size=self.batch_size,
        )

        self.val_set = self.sample_dataset(self.n_samples_val)
        self.val_sampler = SameAttributesSampler(
            attributes=self.val_set.attributes,
            n_samples=self.n_samples_val,
            batch_size=self.batch_size,
        )

        self.test_set = self.sample_dataset(self.n_samples_test)
        self.test_sampler = SameAttributesSampler(
            attributes=self.test_set.attributes,
            n_samples=self.n_samples_test,
            batch_size=self.batch_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_sampler=self.val_sampler,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_sampler=self.test_sampler,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _update_attributes(attributes: np.ndarray, sampler: SameAttributesSampler, dataset: AttributeDataset) -> None:
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.numpy()
        if attributes.ndim == 1:
            attributes = attributes[:, None]

        sampler.attributes = attributes
        dataset.attributes = attributes

    def update_train_attributes(self, attributes: np.ndarray) -> None:
        self._update_attributes(attributes, self.train_sampler, self.train_set)

    def update_val_attributes(self, attributes: np.ndarray) -> None:
        self._update_attributes(attributes, self.val_sampler, self.val_set)

    def update_test_attributes(self, attributes: np.ndarray) -> None:
        self._update_attributes(attributes, self.test_sampler, self.test_set)

