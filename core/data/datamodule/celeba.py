from typing import Optional, Callable, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from core.data.datasets import CelebADict
from core.data.sampler import SameAttributesSampler
from core.data.utils import CompareAttributeSubset


class CelebADataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 num_workers: int,
                 batch_size: int,
                 train_transforms: Optional[Callable] = None,
                 val_transforms: Optional[Callable] = None,
                 train_attributes: Optional[List[int]] = None,
                 sample_same_attributes: bool = False,
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

    def setup(self, stage: Optional[str] = None):
        self.train_set = CelebADict(
            root=self.data_dir,
            transform=self.train_transforms,
            select_attributes=self.train_attributes,
            split="train",
            download=self.download
        )

        self.val_set = CelebADict(
            root=self.data_dir,
            transform=self.val_transforms,
            select_attributes=self.train_attributes,
            split="valid",
            download=self.download)

        # Define a function that compares the attributes of two images
        compare_attributes = CompareAttributeSubset(self.train_attributes)

        if self.sample_same_attributes:
            # Sampler that samples images with the same attributes (subset)
            self.train_sampler = SameAttributesSampler(
                attributes=self.train_set.attr,
                compare_attributes=compare_attributes,
                batch_size=self.batch_size)

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
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


