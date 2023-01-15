from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from core.data.datamodule.base import DataModuleWithAttributes
from core.data.datasets import SampleDataset
from core.data.sampler import SameAttributesSampler
from core.distributions.test import MultivariateCorrelatedNormalMixture, SignResampledDistribution
from core.distributions.test.multivariate_normal_mixture import JointDistribution


class SampledDataModule(DataModuleWithAttributes):
    def __init__(
            self,
            dist: JointDistribution,
            batch_size: int = 64,
            num_workers: int = 1,
            samples_per_epoch: int = 100000,
            store_samples: bool = False,
            min_batch_size: int = 32,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.min_batch_size = min_batch_size

        if num_workers == 0:
            num_workers = 1
        if store_samples:
            samples_per_worker = 0
            self.batch_size = batch_size
            self.collate_fn = None
            self.val_set = SampleDataset(
                dist,
                samples_per_epoch=samples_per_epoch,
                n_samples=samples_per_worker,
                store_samples=store_samples
            )

            self.train_sampler = SameAttributesSampler(
                n_samples=samples_per_epoch,
                batch_size=self.batch_size,
                min_batch_size=self.min_batch_size
            )
            self.val_sampler = SameAttributesSampler(
                n_samples=samples_per_epoch,
                batch_size=self.batch_size,
                min_batch_size=self.min_batch_size
            )

        else:
            assert batch_size % num_workers == 0, "Batch size must be divisible by number of workers"
            samples_per_worker = batch_size // num_workers
            self.batch_size = self.num_workers if self.num_workers > 0 else 1

            def collate_fn(batch: List[Dict[str, torch.Tensor]]):
                c_batch = {k: [] for k in batch[0].keys()}

                for item in batch:
                    for k, v in item.items():
                        c_batch[k].append(v)

                return {k: torch.cat(v, 0) for k, v in c_batch.items()}

            self.collate_fn = collate_fn
            self.val_set = None
        self.train_set = SampleDataset(
            dist,
            samples_per_epoch=samples_per_epoch,
            n_samples=samples_per_worker,
            store_samples=store_samples
        )

    def train_dataloader(self):
        if self.train_sampler is None:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                shuffle=True
            )
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        if self.val_set:
            return DataLoader(
                self.val_set,
                batch_sampler=self.val_sampler,
                num_workers=self.num_workers,
            )

    def update_train_attributes(self, attributes: np.ndarray) -> None:
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.numpy()
        if attributes.ndim == 1:
            attributes = attributes[:, None]

        if self.train_sampler.attributes is None:
            self.train_sampler.attributes = attributes
        else:
            self.train_sampler.attributes = np.concatenate([self.train_sampler.attributes, attributes], axis=1)

        if self.train_set.attributes is None:
            self.train_set.attributes = attributes
        else:
            self.train_set.attributes = np.concatenate([self.train_set.attributes, attributes], axis=1)

    def update_val_attributes(self, attributes: np.ndarray) -> None:
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.numpy()
        if attributes.ndim == 1:
            attributes = attributes[:, None]

        if self.val_sampler.attributes is None:
            self.val_sampler.attributes = attributes
        else:
            self.val_sampler.attributes = np.concatenate([self.val_sampler.attributes, attributes], axis=1)

        if self.val_set.attributes is None:
            self.val_set.attributes = attributes
        else:
            self.val_set.attributes = np.concatenate([self.val_set.attributes, attributes], axis=1)


class SampledNormalMixture(SampledDataModule):
    def __init__(
            self,
            n_dim: int = 5,
            batch_size: int = 128,
            num_workers: int = 1,
            samples_per_epoch: int = 100000,
            neg_samples: int = 1,
            store_samples: bool = False,
            min_batch_size: int = 32,
    ):

        p_xy = MultivariateCorrelatedNormalMixture(n_dim=n_dim)
        p_xya = SignResampledDistribution(p_xy, neg_samples=neg_samples)
        self.p_xya = p_xya

        super().__init__(
            dist=p_xya,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_epoch=samples_per_epoch,
            store_samples=store_samples,
            min_batch_size=min_batch_size,
        )

    @property
    def h_y(self) -> float:
        return self.p_xya.entropy("y")

    @property
    def h_a(self) -> float:
        return self.p_xya.entropy("a")

    @property
    def neg_samples(self) -> int:
        return self.p_xya.neg_samples

    @neg_samples.setter
    def neg_samples(self, value: int):
        if value <= 0:
            value = self.batch_size - value
        self.p_xya.neg_samples = value
