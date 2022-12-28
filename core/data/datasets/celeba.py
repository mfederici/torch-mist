from torchvision.datasets import CelebA
import torch
import numpy as np
from typing import Optional, Callable
# from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform


class CelebADict(CelebA):
    def __init__(self, root: str, select_attributes, transform: Callable, download: bool = False, split: str = "train", augment_x: bool = True):
        # if split == "train":
        #     transform = SimCLRTrainDataTransform(input_height=218)
        # else:
        #     raise NotImplemented()

        super().__init__(root=root, download=download, split=split, transform=transform)
        self.select_attributes = select_attributes
        self.not_selected_attributes = torch.LongTensor([i for i in np.arange(40) if not (i in select_attributes)])
        self.n_attributes = len(select_attributes)
        self.augment_x = augment_x

    @property
    def h_a(self):
        attr = self.attr[:,self.select_attributes]
        c_attr = torch.eye(2**self.n_attributes)[(2**torch.arange(self.n_attributes)*attr).sum(1)].sum(0)
        c_attr_dist = c_attr/c_attr.sum()
        h_a = -(c_attr_dist[c_attr_dist>0]*torch.log(c_attr_dist[c_attr_dist>0])).sum()
        return h_a

    def __getitem__(self, item):
        pos_imgs, a = super().__getitem__(item)

        x = pos_imgs[0 if self.augment_x else 2]
        y = pos_imgs[1]

        return {
            "x": x,
            "y": y,
            "a": a[self.select_attributes],
            "t": a[self.not_selected_attributes]
        }


class ContrastiveCelebA(CelebADict):
    def _find_same_attributes(self, a: torch.LongTensor):
        mask = (self.attr[:,self.select_attributes] == a).sum(1) == len(a)
        ids = torch.arange(len(self))[mask]
        return np.random.choice(ids)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        idx = self._find_same_attributes(data["a"])
        neg_data, a_ = super().__getitem__(idx)
        data["y_"] = neg_data["y"]

        return data

