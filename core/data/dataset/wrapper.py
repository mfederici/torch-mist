from torch.utils.data import Dataset
from torch.distributions import Distribution


class SampleDataset(Dataset):
    def __init__(self, distribution: Distribution, samples_per_epoch: int = 100000, n_samples: int = 0):
        super(SampleDataset, self).__init__()
        self.distribution = distribution
        self.samples_per_epoch = samples_per_epoch
        self.n_samples = n_samples

    def __getitem__(self, item):
        if self.n_samples > 0:
            samples = self.distribution.sample([self.n_samples])
        else:
            samples = self.distribution.sample()
        return samples

    def __len__(self):
        if self.n_samples == 0:
            return self.samples_per_epoch
        else:
            return self.samples_per_epoch//self.n_samples
