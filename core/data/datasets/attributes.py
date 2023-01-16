from typing import Optional

import torch
from torch.utils.data import Dataset
import numpy as np


class AttributeDataset(Dataset):
    def __init__(self, dataset: Dataset, attributes: Optional[torch.LongTensor] = None):
        super().__init__()

        self.dataset = dataset
        if attributes is not None:
            n_attributes = np.max(attributes)+1
        else:
            n_attributes = 0
        self.n_attributes = n_attributes
        self._attributes = attributes
        self._h_a = None

    def __getitem__(self, item):
        data = self.dataset[item]
        if self.attributes is not None:
            data['a'] = self.attributes[item]

        return data

    def __len__(self):
        return len(self.dataset)

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        self._attributes = value
        self._h_a = None

    def _compute_attribute_entropy(self):
        if self.attributes is None:
            return 0
        else:
            base = np.max(self.attributes) + 1
            c_attr, _ = np.histogram((base ** torch.arange(self.attributes.shape[1]) * self.attributes).sum(1),
                                     bins=np.arange(base ** self.n_attributes + 1))
            c_attr_dist = c_attr / c_attr.sum()
            h_a = -np.sum(c_attr_dist[c_attr_dist > 0] * np.log(c_attr_dist[c_attr_dist > 0]))
            self._h_a = h_a
    @property
    def h_a(self):
        if self._h_a is None:
            self._compute_attribute_entropy()
        return self._h_a