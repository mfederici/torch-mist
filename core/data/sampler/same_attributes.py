from torch.utils.data import Sampler
from typing import Callable
import torch
import numpy as np





class SameAttributesSampler(Sampler):
    def __init__(self, attributes: torch.Tensor, compare_attributes: Callable[[torch.Tensor, torch.Tensor], torch.BoolTensor], batch_size: int):
        super(SameAttributesSampler, self).__init__(attributes)
        self.compare_attributes = compare_attributes
        self.attributes = attributes
        self.batch_size = batch_size

    def __iter__(self):
        mask = torch.ones(len(self.attributes)).bool()
        all_ids = torch.arange(len(self.attributes))

        while mask.sum() > 0:
            a_idx = np.random.choice(all_ids[mask],1)
            a = self.attributes[a_idx].squeeze(0)
            viable_ids = all_ids[self.compare_attributes(self.attributes, a)]
            if self.batch_size-1 <= len(viable_ids):
                batch_ids = np.random.choice(viable_ids, self.batch_size-1, replace=False)
            else:
                batch_ids = np.random.choice(torch.cat([viable_ids, torch.LongTensor(a_idx)], 0), self.batch_size-1, replace=True)

            batch_ids = np.concatenate([batch_ids, a_idx], 0)
            mask[batch_ids] = False
            yield list(batch_ids)

    def __len__(self):
        return self.attributes.shape[0] // self.batch_size
