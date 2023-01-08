from torchvision.datasets import CelebA
import torch
import numpy as np
from typing import Optional, Callable, List, Tuple, Any, Dict


# from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform


class CelebADict(CelebA):
    def __init__(
            self,
            root: str,
            transform: Callable[[Any], torch.Tensor],
            select_attributes: Optional[List[int]] = None,
            download: bool = False,
            split: str = "train",
            augment_x: bool = True,

    ):

        super().__init__(root=root, download=download, split=split)
        if select_attributes is None:
            select_attributes = list(range(40))
        self.transform_x = transform
        self.transform_y = transform
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

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        img, a = super().__getitem__(item)
        x = self.transform_x(img)
        y = self.transform_y(img)

        return {
            "x": x,
            "y": y,
            "a": a[self.select_attributes],
            "t": a[self.not_selected_attributes]
        }


class ContrastiveCelebA(CelebADict):
    def __init__(
        self,
        root: str,
        transform: Callable[[Any], torch.Tensor],
        select_attributes: Optional[List[int]] = None,
        download: bool = False,
        split: str = "train",
        augment_x: bool = True,
        neg_samples: int = 1,
        sample_negatives: bool = True,
    ):
        super().__init__(root, transform, select_attributes, download, split, augment_x)
        self.sample_negatives = sample_negatives
        self.neg_samples = neg_samples
        self._id_cache = {}

    def _find_same_attributes(self, a: torch.Tensor):
        if a in self._id_cache:
            return self._id_cache[a]
        else:
            mask = (self.attr[:, self.select_attributes] == a).sum(1) == len(a)
            ids = torch.arange(len(self))[mask]
            self._id_cache[a] = ids
        return np.random.choice(ids, self.neg_samples)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if self.neg_samples > 0 and self.sample_negatives:
            assert "a" in data, "a not in data"
            y_ = []
            a = data["a"]
            for idx in self._find_same_attributes(a):
                neg_img, a_ = CelebA.__getitem__(self, idx)
                assert torch.equal(a, a_[self.select_attributes]), f"{a} != {a_}, a and a_ should be equal"
                neg_sample = self.transform_y(neg_img)
                y_.append(neg_sample.unsqueeze(0))

            y_ = torch.cat(y_, 0)
            if self.neg_samples == 1:
                y_ = y_.squeeze(0)
            data['y_'] = y_

        return data

