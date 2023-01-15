from torch.utils.data import Dataset
import torch

from core.distributions.test.multivariate_normal_mixture import JointDistribution


class SampleDataset(Dataset):
    def __init__(
            self,
            distribution: JointDistribution,
            samples_per_epoch: int = 100000,
            n_samples: int = 0,
            store_samples: bool = False,
            attributes=None,
    ):
        super(SampleDataset, self).__init__()

        self.distribution = distribution
        self.samples_per_epoch = samples_per_epoch
        self.n_samples = n_samples
        if store_samples:
            self.samples = self.distribution.sample(torch.Size([samples_per_epoch]))
        else:
            self.samples = None

        assert not store_samples or attributes is None, "Attributes not supported for stored samples"
        assert attributes is None or len(
            attributes) == samples_per_epoch, "Attributes must be of length samples_per_epoch"

        self.attributes = attributes

    def __getitem__(self, idx):
        if self.samples is not None:
            data = {k: v[idx] for k, v in self.samples.items()}
            if self.attributes is not None:
                data["a"] = self.attributes[idx]
            data['idx'] = idx
            return data
        else:
            if self.n_samples > 0:
                samples = self.distribution.sample(torch.Size([self.n_samples]))
            else:
                samples = self.distribution.sample()
            return samples

    def __len__(self):
        if self.n_samples == 0:
            return self.samples_per_epoch
        else:
            return self.samples_per_epoch//self.n_samples
