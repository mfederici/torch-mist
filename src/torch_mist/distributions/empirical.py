import torch
from torch.distributions import Distribution
from torch_mist.utils.indexing import select_off_diagonal


class EmpiricalDistribution(Distribution):
    def __init__(self):
        super().__init__(validate_args=False)
        self._samples = None

    def add_samples(self, samples):
        self._samples = samples

    def sample(self, sample_shape=torch.Size()):
        assert self._samples is not None
        assert len(sample_shape) == 1
        n_samples = sample_shape[0]

        return select_off_diagonal(self._samples, n_samples)

    def update(self):
        self._samples = None
