from torch.utils.data import Sampler
from typing import Union, Optional
import torch
import numpy as np
from collections import Counter


class SameAttributeSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        neg_samples: int,
        attributes: Union[torch.Tensor, np.ndarray],
    ):
        super(SameAttributeSampler, self).__init__(attributes)

        # Whenever the number of negatives is specified as 0 or negative, we produce whole batches of negatives
        if neg_samples <= 0:
            neg_samples = batch_size - 1

        assert (
            batch_size % (neg_samples + 1) == 0
        ), "batch_size has to be a multiple of neg_samples+1"

        assert attributes is not None
        if isinstance(attributes, torch.Tensor):
            attributes = attributes.data.numpy()
        if attributes.ndim == 2:
            assert attributes.shape[1] == 1
            attributes = attributes.reshape(-1)
        attributes = attributes.astype(np.int32)

        assert min(attributes) == 0, "Attributes should start from 0"
        counts = np.bincount(attributes, minlength=max(attributes))
        for attribute, count in enumerate(counts):
            assert (
                count >= neg_samples
            ), f"The attribute count for {attribute} is less than the number of negatives, please use neg_samples<={count}"

        self.n_attributes = max(attributes) + 1
        self.neg_samples = neg_samples + 1
        self.batch_size = batch_size
        self.chunks_per_batch = self.batch_size // self.neg_samples
        self.a_ids = {}
        self.n_chunks = []
        for a in range(self.n_attributes):
            a_ids = np.arange(len(attributes))[attributes == a]
            self.a_ids[a] = a_ids
            self.n_chunks.append(a_ids.shape[0] // self.neg_samples)

        self.chunked_attributes = []
        for a, n in enumerate(self.n_chunks):
            self.chunked_attributes += [a] * n

        self._len = None

    #
    # def make_batch(self, shuffled_a_ids, n_valid) -> np.ndarray:
    #     # Determine which attributes are included in the batch
    #     att_samples = np.random.choice(n_valid.sum(), self.batch_size // self.neg_samples, replace=False)
    #     cdf = np.cumsum(n_valid)
    #     sampled_atts = list(np.sum(att_samples.reshape(-1, 1) >= cdf.reshape(1, -1), -1))
    #     att_counts = Counter(sampled_atts)
    #
    #     ids = []
    #     for a, count in att_counts.items():
    #         if n_valid[a] == count:
    #             abs_ids = shuffled_a_ids[a][-n_valid[a]:]
    #         else:
    #             abs_ids = shuffled_a_ids[a][-n_valid[a]:-n_valid[a]+count]
    #
    #         n_valid[a] -= count
    #         # Add them to the sampled ids
    #         if len(ids) == 0:
    #             ids = abs_ids
    #         else:
    #             ids = np.concatenate([ids, abs_ids], 0)
    #     return ids.reshape(-1, self.neg_samples).T.reshape(-1)

    def __iter__(self):
        shuffled_a_ids = {}
        # Shuffle the ids for each attributes
        for a, att_ids in self.a_ids.items():
            permutation = np.random.permutation(att_ids)
            # Keep multiples of neg_samples
            permutation = permutation[
                : (permutation.shape[0] // self.neg_samples) * self.neg_samples
            ]
            shuffled_a_ids[a] = permutation.reshape(
                -1,
                self.neg_samples,
            )

        shuffled_a_chunks = np.random.permutation(self.chunked_attributes)
        shuffled_a_chunks = shuffled_a_chunks[
            : (len(shuffled_a_chunks) // self.chunks_per_batch)
            * self.chunks_per_batch
        ]
        shuffled_a_chunks = shuffled_a_chunks.reshape(
            -1, self.chunks_per_batch
        )

        id_pointers = {a: 0 for a in shuffled_a_ids}
        for chunk in shuffled_a_chunks:
            batch_ids = []
            for a in chunk:
                batch_ids += list(shuffled_a_ids[a][id_pointers[a]])
                id_pointers[a] += 1

            yield np.array(batch_ids).reshape(-1, self.neg_samples).T.reshape(
                -1
            )

    def __len__(self):
        n_neg_samples = 0
        for a, atts in self.a_ids.items():
            n_neg_samples += len(atts) // self.neg_samples

        n_batches = n_neg_samples // (self.batch_size // self.neg_samples)
        return n_batches
