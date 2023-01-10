from torch.utils.data import Sampler
from typing import Union
import torch
import numpy as np


class SameAttributesSampler(Sampler):
    def __init__(
            self,
            attributes: Union[torch.Tensor, np.ndarray],
            batch_size: int,
            fixed_batch_size: bool = False,
            build_cache: bool = True,
            min_batch_size: int = 0,
    ):
        super(SameAttributesSampler, self).__init__(attributes)
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.data.numpy()
        if attributes.ndim == 1:
            attributes = attributes.reshape(-1, 1)
        self.attributes = attributes
        self.batch_size = batch_size
        self.fixed_batch_size = fixed_batch_size
        self.all_ids = np.arange(self.attributes.shape[0])
        self._len = None
        self.build_cache = build_cache
        self._cache = dict()
        self.min_batch_size = min_batch_size

    def _compute_len(self):
        mask = np.ones(len(self.attributes)) == 1
        l = 0
        while mask.sum() > 0:
            batch_ids = self.make_batch(mask)
            # Make sure the selected ids are not re-sampled for this epoch
            mask[batch_ids] = False
            if len(batch_ids) >= self.min_batch_size:
                l += 1
        self._len = l

    def compare_attributes(self, a):
        a_key = a.tobytes()
        if not (a_key in self._cache):
            v = np.sum(self.attributes == a, -1) == self.attributes.shape[-1]
            self._cache[a_key] = v
        return self._cache[a_key]

    def make_batch(self, mask: np.ndarray) -> np.ndarray:
        # Pick a random id that has not been sampled yet
        a_idx = np.random.choice(self.all_ids[mask], 1)[0]

        a = self.attributes[a_idx]

        # Consider all the ids with the same attribute that have not been sampled
        a_mask = self.compare_attributes(a)
        viable_ids = self.all_ids[a_mask & mask]

        # If the number of available ids is at least as big as the batch size, select batch_size of them at random
        if self.batch_size <= len(viable_ids):
            batch_ids = np.random.choice(viable_ids, self.batch_size, replace=False)
        else:
            # Otherwise, if fixed_batch_size is true, select all the ids that are available with repetition
            if self.fixed_batch_size:
                batch_ids = np.random.choice(
                    torch.cat([viable_ids, torch.LongTensor(a_idx)], 0),
                    self.batch_size,
                    replace=True
                )
            else:
                # Otherwise, select all the ids that are available without repetition (smaller batch)
                batch_ids = viable_ids

        return batch_ids

    def __iter__(self):
        mask = np.ones(len(self.attributes)) == 1

        while mask.sum() > 0:
            batch_ids = self.make_batch(mask)
            # Make sure the selected ids are not re-sampled for this epoch
            mask[batch_ids] = False
            # Discard batches that are too small
            if len(batch_ids) >= self.min_batch_size:
                yield batch_ids

    def __len__(self):
        if self._len is None:
            self._compute_len()
        return self._len
