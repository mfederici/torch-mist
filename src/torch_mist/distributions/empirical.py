import torch
from torch.distributions import Distribution

from torch_mist.utils.indexing import select_k_others


class EmpiricalDistribution(Distribution):
    def __init__(self):
        super().__init__(validate_args=False)
        self._samples = None

    def add_samples(self, samples):
        self._samples = samples

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        assert len(sample_shape) == 1
        n_samples = sample_shape[0]
        return select_k_others(self._samples, n_samples)

    def update(self):
        self._samples = None

    def __repr__(self):
        return f"{self.__class__.__name__}(n_samples={None if self._samples is None else len(self._samples)})"
