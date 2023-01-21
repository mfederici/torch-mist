from torch.utils.data import Dataset
import torch

from core.distributions.test.multivariate_normal_mixture import JointDistribution


class SampleDataset(Dataset):
    def __init__(
            self,
            distribution: JointDistribution,
            n_samples: int = 100000,
    ):
        super(SampleDataset, self).__init__()

        self.distribution = distribution
        self.samples = self.distribution.sample(torch.Size([n_samples]))

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.samples.items()}
        data['idx'] = idx
        return data

    def __len__(self):
        len(self.samples)
