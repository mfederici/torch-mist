from typing import Optional, Callable, List, Dict, Union, Any

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from core.data.datasets import MultiOmniglot
from core.data.datasets.attributes import AttributeDataset
from core.data.sampler import SameAttributesSampler
from core.data.datamodule.base import DataModuleWithAttributes
import os

OMNIGLOT_DIR = "OMNIGLOT"

class MultiOmniglotDataModule(DataModuleWithAttributes):
    def __init__(self,
                 data_dir: str,
                 num_workers: int,
                 batch_size: int,
                 n_images: int = 1,
                 download: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.n_images = n_images
        self.download = download

    @property
    def h_a(self):
        return self.train_set.h_a

    def setup(self, stage: Optional[str] = None):
        self.train_set = AttributeDataset(
            MultiOmniglot(
                root=os.path.join(self.data_dir, OMNIGLOT_DIR),
                n_images=self.n_images,
                split="train",
                download=self.download
            )
        )

        self.val_set = AttributeDataset(
            MultiOmniglot(
                root=os.path.join(self.data_dir, OMNIGLOT_DIR),
                n_images=self.n_images,
                split="val",
                download=self.download
            )
        )

        # Sampler that samples images with the same attributes (subset)
        self.train_sampler = SameAttributesSampler(
            attributes=None,
            n_samples=len(self.train_set),
            batch_size=self.batch_size,
            min_batch_size=0
        )

        self.val_sampler = SameAttributesSampler(
            attributes=None,
            n_samples=len(self.val_set),
            batch_size=self.batch_size,
            min_batch_size=0
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_sampler=self.val_sampler,
            num_workers=self.num_workers
        )

    def update_train_attributes(self, attributes: np.ndarray) -> None:
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.numpy()
        if attributes.ndim == 1:
            attributes = attributes[:, None]

        self.train_sampler.attributes = attributes
        self.train_set.attributes = attributes

    def update_val_attributes(self, attributes: np.ndarray) -> None:
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.numpy()
        if attributes.ndim == 1:
            attributes = attributes[:, None]

        self.val_sampler.attributes = attributes
        self.val_set.attributes = attributes



