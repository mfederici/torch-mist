import numpy as np
import torch
from torch.utils.data import DataLoader

from tests.data.datasets import MultiOmniglot
from tests.data.datasets.attributes import AttributeDataset
from tests.data.sampler import SameAttributesSampler
from tests.data.datamodule.base import DataModuleWithAttributes
import os

OMNIGLOT_DIR = "OMNIGLOT"


class MultiOmniglotDataModule(DataModuleWithAttributes):
    def __init__(self,
                 data_dir: str,
                 num_workers: int,
                 batch_size: int,
                 n_images: int = 1,
                 train_split="train",
                 val_split="val",
                 test_split="test",
                 n_train_samples: int = 50000,
                 n_val_samples: int = 10000,
                 n_test_samples: int = 10000,
                 download: bool = False
    ):
        super().__init__()
        if data_dir.startswith("$"):
            data_dir = os.path.expandvars(data_dir)

        print("READING FROM", data_dir)

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.n_images = n_images
        self.download = download
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    @property
    def h_a(self):
        return self.train_set.h_a

    def prepare_data(self):
        # Train
        self.train_set = AttributeDataset(
            MultiOmniglot(
                root=os.path.join(self.data_dir, OMNIGLOT_DIR),
                n_images=self.n_images,
                n_samples=self.n_train_samples,
                split=self.train_split,
                download=self.download
            )
        )

        # Sampler that samples images with the same attributes (subset)
        self.train_sampler = SameAttributesSampler(
            attributes=None,
            n_samples=len(self.train_set),
            batch_size=self.batch_size,
        )

        # Validation set
        self.val_set = AttributeDataset(
            MultiOmniglot(
                root=os.path.join(self.data_dir, OMNIGLOT_DIR),
                n_images=self.n_images,
                n_samples=self.n_val_samples,
                split=self.val_split,
                download=self.download
            )
        )

        self.val_sampler = SameAttributesSampler(
            attributes=None,
            n_samples=len(self.val_set),
            batch_size=self.batch_size,
        )

        # Test set
        self.test_set = AttributeDataset(
            MultiOmniglot(
                root=os.path.join(self.data_dir, OMNIGLOT_DIR),
                n_images=self.n_images,
                n_samples=self.n_test_samples,
                split=self.test_split,
                download=self.download
            )
        )

        self.test_sampler = SameAttributesSampler(
            attributes=None,
            n_samples=len(self.test_set),
            batch_size=self.batch_size,
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

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_sampler=self.test_sampler,
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

    def update_test_attributes(self, attributes: np.ndarray) -> None:
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.numpy()
        if attributes.ndim == 1:
            attributes = attributes[:, None]

        self.test_sampler.attributes = attributes
        self.test_set.attributes = attributes
