import pytorch_lightning as pl
import torch

from typing import List, Dict
from torch.utils.data import DataLoader

from core.data.datasets import SampleDataset
from core.distributions.test import MultivariateCorrelatedNormalMixture, SignResampledDistribution
from core.distributions.test.multivariate_normal_mixture import JointDistribution


def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    c_batch = {k: [] for k in batch[0].keys()}

    for item in batch:
        for k, v in item.items():
            c_batch[k].append(v)

    return {k: torch.cat(v, 0) for k, v in c_batch.items()}


class SampledDataModule(pl.LightningDataModule):
    def __init__(self, dist: JointDistribution, batch_size: int = 128, num_workers: int = 1, samples_per_epoch: int = 100000):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        if num_workers == 0:
            num_workers = 1
        assert batch_size % num_workers == 0, "Batch size must be divisible by number of workers"
        samples_per_worker = batch_size // num_workers
        self.dataset = SampleDataset(dist, samples_per_epoch=samples_per_epoch, n_samples=samples_per_worker)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.num_workers if self.num_workers > 0 else 1,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )


class SampledNormalMixture(SampledDataModule):
    def __init__(self, batch_size: int = 128, num_workers: int = 1, samples_per_epoch: int = 100000, neg_samples: int = 1):
        p_xy = MultivariateCorrelatedNormalMixture()
        self.batch_size = batch_size
        p_xya = SignResampledDistribution(p_xy, neg_samples=neg_samples)
        self.p_xya = p_xya

        super().__init__(
            dist=p_xya,
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_epoch=samples_per_epoch
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
