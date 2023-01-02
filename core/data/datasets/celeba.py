from torchvision.datasets import CelebA
import torch
import numpy as np
from typing import Optional, Callable, List, Tuple, Any

# from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform


class CelebADict(CelebA):
    def __init__(
            self,
             root: str,
             transform: Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
             select_attributes: Optional[List[int]] = None,
             download: bool = False,
             split: str = "train",
             augment_x: bool = True
    ):

        super().__init__(root=root, download=download, split=split, transform=transform)
        if select_attributes is None:
            select_attributes = list(range(40))
        self.select_attributes = select_attributes
        self.not_selected_attributes = torch.LongTensor([i for i in np.arange(40) if not (i in select_attributes)])
        self.n_attributes = len(select_attributes)
        self.augment_x = augment_x
        self._h_a = None

    def _compute_attribute_entropy(self):
        attr = self.attr[:, self.select_attributes]
        c_attr, _ = np.histogram((2 ** torch.arange(self.n_attributes) * attr).sum(1),
                                 bins=np.arange(2 ** self.n_attributes + 1))
        c_attr_dist = c_attr / c_attr.sum()
        h_a = -np.sum(c_attr_dist[c_attr_dist > 0] * np.log(c_attr_dist[c_attr_dist > 0]))
        self._h_a = h_a

    @property
    def h_a(self):
        if self._h_a is None:
            self._compute_attribute_entropy()
        return self._h_a

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

