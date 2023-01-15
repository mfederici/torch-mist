from typing import Optional, Callable, List, Dict, Union, Any

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from core.data.datasets import CelebADict, ContrastiveCelebA
from core.data.sampler import SameAttributesSampler
from core.data.utils import CompareAttributeSubset
from core.data.datamodule.base import DataModuleWithAttributes


class CelebABatchDataModule(DataModuleWithAttributes):
    def __init__(self,
                 data_dir: str,
                 num_workers: int,
                 batch_size: int,
                 train_transforms: Union[Dict[str, Callable[[Any], torch.Tensor]], Callable[[Any], torch.Tensor]],
                 val_transforms: Union[Dict[str, Callable[[Any], torch.Tensor]], Callable[[Any], torch.Tensor]],
                 train_attributes: Optional[List[int]] = None,
                 weak_supervision: bool = True,
                 sample_same_attributes: bool = False,
                 min_batch_size: int = 0,
                 download: bool = False
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.download = download
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_attributes = train_attributes
        self.sample_same_attributes = sample_same_attributes
        self.min_batch_size = min_batch_size
        self.weak_supervision = weak_supervision

    @property
    def h_a(self):
        return self.train_set.h_a

    def setup(self, stage: Optional[str] = None):
        self.train_set = CelebADict(
            root=self.data_dir,
            transforms=self.train_transforms,
            train_attributes=self.train_attributes,
            weak_supervision=self.weak_supervision,
            split="train",
            download=self.download
        )

        self.val_set = CelebADict(
            root=self.data_dir,
            transforms=self.val_transforms,
            train_attributes=self.train_attributes,
            weak_supervision=self.weak_supervision,
            split="valid",
            download=self.download)

        if self.sample_same_attributes:
            if self.weak_supervision:
                train_attr = self.train_set.attr[:, self.train_attributes]
                val_attr = self.val_set.attr[:, self.train_attributes]
            else:
                train_attr = None
                val_attr = None
            n_train_samples = len(self.train_set)
            n_val_samples = len(self.val_set)

            # Sampler that samples images with the same attributes (subset)
            self.train_sampler = SameAttributesSampler(
                attributes=train_attr,
                n_samples=n_train_samples,
                batch_size=self.batch_size,
                min_batch_size=self.min_batch_size
            )

            self.val_sampler = SameAttributesSampler(
                attributes=val_attr,
                n_samples=n_val_samples,
                batch_size=self.batch_size,
                min_batch_size=self.min_batch_size
            )

        print(f"TrainSampler.attributes: {self.train_sampler.attributes}")

    def train_dataloader(self):
        if self.sample_same_attributes:
            return DataLoader(
                self.train_set,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )

    def val_dataloader(self):
        if self.sample_same_attributes:
            return DataLoader(
                self.val_set,
                batch_sampler=self.val_sampler,
                num_workers=self.num_workers
            )
        else:
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers
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


class ContrastiveCelebADataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str,
            num_workers: int,
            batch_size: int,
            train_transforms: Union[Dict[str, Callable[[Any], torch.Tensor]], Callable[[Any], torch.Tensor]],
            val_transforms: Union[Dict[str, Callable[[Any], torch.Tensor]], Callable[[Any], torch.Tensor]],
            neg_samples: int = 0,
            train_attributes: Optional[List[int]] = None,
            download: bool = False,

    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.train_attributes = train_attributes
        self.download = download
        self._neg_samples = neg_samples

    @property
    def neg_samples(self) -> int:
        return self._neg_samples

    @neg_samples.setter
    def neg_samples(self, value: int):
        self._neg_samples = value
        self.train_set.neg_samples = value
        self.val_set.neg_samples = value

    def prepare_data(self, stage: Optional[str] = None):
        self.train_set = ContrastiveCelebA(
            root=self.data_dir,
            transforms=self.train_transforms,
            train_attributes=self.train_attributes,
            split="train",
            download=self.download,
            neg_samples=self.neg_samples
        )

        self.val_set = ContrastiveCelebA(
            root=self.data_dir,
            transforms=self.val_transforms,
            train_attributes=self.train_attributes,
            split="valid",
            download=self.download,
            neg_samples=self.neg_samples
        )

    @property
    def h_a(self):
        return self.train_set.h_a

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


