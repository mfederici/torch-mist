import pytorch_lightning as pl
import torch

from typing import List, Dict
from torch.distributions import Distribution
from torch.utils.data import DataLoader

from core.data.dataset import SampleDataset
from core.distributions.test import MultivariateCorrelatedNormalMixture, SignResampledDistribution
from core.distributions.test.multivariate_normal_mixture import JointDistribution


def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    c_batch = {k: [] for k in batch[0].keys()}

    for item in batch:
        for k, v in item.items():
            c_batch[k].append(v)

    return {k: torch.cat(v, 0) for k, v in c_batch.items()}


class SampledDataModule(pl.LightningDataModule):
    def __init__(self, dist: JointDistribution, batch_size: int = 128, num_workers: int = 1, n_negatives: int = 1,
                 samples_per_epoch: int = 100000):
        super().__init__()

        self.batch_size = batch_size
        assert batch_size % num_workers == 0, "Batch size must be divisible by number of workers"
        self.num_workers = num_workers
        samples_per_worker = batch_size // num_workers
        self.n_negatives = n_negatives
        self.dataset = SampleDataset(dist, samples_per_epoch=samples_per_epoch, n_samples=samples_per_worker)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.num_workers,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )


class SampledNormalMixture(SampledDataModule):
    def __init__(self, batch_size: int = 128, num_workers: int = 1, samples_per_epoch: int = 100000):
        p_xy = MultivariateCorrelatedNormalMixture()

        p_xya = SignResampledDistribution(p_xy)
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


