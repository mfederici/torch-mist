from torchvision.datasets import CelebA
import torch
import numpy as np
from typing import Optional, Callable, List, Tuple, Any, Dict, Union


# from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform


class CelebADict(CelebA):
    def __init__(
            self,
            root: str,
            transforms: Union[Dict[str, Callable[[Any], torch.Tensor]], Callable[[Any], torch.Tensor]],
            train_attributes: Optional[List[int]] = None,
            download: bool = False,
            weak_supervision: bool = True,
            split: str = "train",
            augment_x: bool = True,

    ):

        super().__init__(root=root, download=download, split=split)
        if train_attributes is None:
            train_attributes = list(range(40))
        if not isinstance(transforms, dict):
            transforms = {'x': transforms, 'y': transforms}
        self.transforms = transforms

        self.train_attributes = train_attributes
        self.not_selected_attributes = torch.LongTensor([i for i in np.arange(40) if not (i in train_attributes)])

        if weak_supervision:
            self._attributes = self.attr[:, self.train_attributes].data.numpy()
        else:
            self._attributes = None
        self.test_attributes = self.attr[:, self.not_selected_attributes].data.numpy()

        self.n_attributes = len(train_attributes)
        self.augment_x = augment_x
        self._h_a = None

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

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        img, a = super().__getitem__(item)

        data = {}
        for key, transform in self.transforms.items():
            data[key] = transform(img)

        if self.attributes is not None:
            data['a'] = self.attributes[item]

        data['t'] = self.test_attributes[item]
        data['idx'] = item

        return data


class ContrastiveCelebA(CelebADict):
    def __init__(
        self,
        root: str,
        transforms: Dict[str, Callable[[Any], torch.Tensor]],
        train_attributes: Optional[List[int]] = None,
        download: bool = False,
        split: str = "train",
        augment_x: bool = True,
        neg_samples: int = 1,
        weak_supervision: bool = True,
    ):
        super().__init__(
            root=root,
            transforms=transforms,
            train_attributes=train_attributes,
            download=download,
            split=split,
            augment_x=augment_x,
            weak_supervision=weak_supervision,
        )

        assert "y" in self.transforms, "y not in transforms"
        self.neg_samples = neg_samples
        self.select_attributes = train_attributes
        self.all_ids = np.arange(len(self))
        self._cache = {}

    def _find_same_attributes(self, a: torch.Tensor):
        a_key = a.numpy().tobytes()
        if not (a_key in self._cache):
            v = torch.sum(self.attr[:, self.select_attributes] == a, -1) == len(self.select_attributes)
            ids = self.all_ids[v]
            self._cache[a_key] = ids
        else:
            ids = self._cache[a_key]

        return np.random.choice(ids, self.neg_samples)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if self.neg_samples > 0:
            assert "a" in data, "a not in data"
            y_ = []
            a = data["a"]

            neg_ids = self._find_same_attributes(a)
            assert len(neg_ids) == self.neg_samples
            for idx in neg_ids:
                neg_img, a_ = CelebA.__getitem__(self, idx)
                assert torch.equal(a, a_[self.select_attributes]), f"{a} != {a_}, a and a_ should be equal"
                neg_sample = self.transforms['y'](neg_img)
                y_.append(neg_sample.unsqueeze(0))

            y_ = torch.cat(y_, 0)
            if self.neg_samples == 1:
                y_ = y_.squeeze(0)
            data['y_'] = y_

        return data

