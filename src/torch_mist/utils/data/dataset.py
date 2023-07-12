from typing import Dict, Sequence

import torch

class SampleDataset(Sequence):
    def __init__(self, samples: Dict[str, torch.Tensor]):
        self.samples = samples
        # Check all the samples have the same length
        lengths = [len(value) for value in samples.values()]
        assert all([length == lengths[0] for length in lengths])
        self.n_samples = lengths[0]

    def __getitem__(self, item):
        return {name: value[item] for name, value in self.samples.items()}

    def __len__(self):
        return self.n_samples