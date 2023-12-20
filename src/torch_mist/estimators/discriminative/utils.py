from torch_mist.utils.indexing import select_off_diagonal


class SampleBuffer:
    def __init__(self):
        self._samples = None

    def add_samples(self, samples):
        self._samples = samples

    def sample(self, n_samples: int):
        return select_off_diagonal(self._samples, n_samples)

    def update(self):
        self._samples = None

    def __repr__(self):
        return f"{self.__class__.__name__}(n_samples={None if self._samples is None else len(self._samples)})"
