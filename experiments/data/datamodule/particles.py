import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from tests.data.datamodule.base import DataModuleWithAttributes
from tests.data.datasets.particles import ParticleTrajectories
from tests.data.datasets.attributes import AttributeDataset
from tests.data.sampler import SameAttributesSampler


class ParticlesDataModule(DataModuleWithAttributes):
    def __init__(
            self,
            data_dir: str,
            traj_file: str = "correlated_clusters.npy",
            attribute_file: Optional[str] = None,
            n_particles: Optional[int] = None,
            train_on: str = "all",
            val_on: str = "val",
            test_on: str = "test",
            batch_size: int = 128,
            num_workers: int = 12,
    ):
        super().__init__()

        if data_dir.startswith("$"):
            data_dir = os.path.expandvars(data_dir)

        self.data_dir = data_dir
        self.traj_file = traj_file
        self.attribute_file = attribute_file
        self.n_particles = n_particles
        self.train_on = train_on
        self.val_on = val_on
        self.test_on = test_on
        self.num_workers = num_workers
        self.batch_size = batch_size
    def prepare_data(self) -> None:
        self.train_set = ParticleTrajectories(
            data_dir=self.data_dir,
            n_particles=self.n_particles,
            traj_file=self.traj_file,
            attribute_file=self.attribute_file,
            split=self.train_on,
        )
        self.train_sampler = SameAttributesSampler(
            attributes=self.train_set.attributes,
            n_samples=len(self.train_set),
            batch_size=self.batch_size,
        )

        self.val_set = ParticleTrajectories(
            data_dir=self.data_dir,
            n_particles=self.n_particles,
            traj_file=self.traj_file,
            attribute_file=self.attribute_file,
            split=self.val_on,
        )
        self.val_sampler = SameAttributesSampler(
            attributes=self.val_set.attributes,
            n_samples=len(self.val_set),
            batch_size=self.batch_size,
        )

        self.test_set = ParticleTrajectories(
            data_dir=self.data_dir,
            n_particles=self.n_particles,
            traj_file=self.traj_file,
            attribute_file=self.attribute_file,
            split=self.test_on,
        )
        self.test_sampler = SameAttributesSampler(
            attributes=self.test_set.attributes,
            n_samples=len(self.test_set),
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

