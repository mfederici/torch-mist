from torch.utils.data import Sampler
from typing import Union, Optional
import torch
import numpy as np


class SameAttributeSampler(Sampler):
    def __init__(
            self,
            batch_size: int,
            attributes: Optional[Union[torch.Tensor, np.ndarray]] = None,
            n_samples: Optional[int] = None,
            min_batch_size: int = 0,
    ):
        super(SameAttributeSampler, self).__init__(attributes)

        if attributes is not None and n_samples is not None:
            assert len(attributes) == n_samples, "Attributes must be of length n_samples"
        assert attributes is not None or n_samples is not None, "Either attributes or n_samples must be provided"
        if attributes is not None:
            if isinstance(attributes, torch.Tensor):
                attributes = attributes.data.numpy()
            if attributes.ndim == 1:
                attributes = attributes.reshape(-1, 1)
        if n_samples is None:
            n_samples = len(attributes)

        self.n_samples = n_samples

        self._attributes = attributes
        self.batch_size = batch_size
        self.all_ids = np.arange(n_samples)
        self._len = None
        self._cache = dict()
        self.min_batch_size = min_batch_size

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = attributes
        self._cache = dict()
        self._len = None

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
        if isinstance(a, torch.Tensor):
            a = a.data.numpy()
        a_key = a.tobytes()
        if not (a_key in self._cache):
            v = np.sum(self.attributes == a, -1) == self.attributes.shape[-1]
            self._cache[a_key] = v
        return self._cache[a_key]

    def make_batch(self, mask: np.ndarray) -> np.ndarray:
        if self.attributes is not None:
            # Pick a random id that has not been sampled yet
            a_idx = np.random.choice(self.all_ids[mask], 1)[0]

            a = self.attributes[a_idx]

            # Consider all the ids with the same attribute that have not been sampled
            a_mask = self.compare_attributes(a)
            mask = mask & a_mask

        viable_ids = self.all_ids[mask]

        # If the number of available ids is at least as big as the batch size, select batch_size of them at random
        if self.batch_size <= len(viable_ids):
            batch_ids = np.random.choice(viable_ids, self.batch_size, replace=False)
        else:
            # Otherwise, select all the ids that are available without repetition (smaller batch)
            batch_ids = viable_ids

        return batch_ids

    def __iter__(self):
        mask = np.ones(self.n_samples) == 1

        while mask.sum() > 0:
            batch_ids = self.make_batch(mask)
            # Make sure the selected ids are not re-sampled for this epoch
            mask[batch_ids] = False
            # Discard batches that are too small
            if len(batch_ids) >= self.min_batch_size:
                yield batch_ids

    def __len__(self):
        if self.attributes is None:
            extra_batch = 1 if self.n_samples % self.batch_size >= self.min_batch_size else 0
            return self.n_samples // self.batch_size + extra_batch

        if self._len is None:
            self._compute_len()
        return self._len