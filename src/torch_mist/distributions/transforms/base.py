from abc import ABC
from typing import List, Union, Dict

import torch
from torch import nn

from pyro.distributions import (
    ConditionalDistribution,
    ConditionalTransform,
    ConditionalTransformedDistribution,
)
from torch.distributions import (
    Distribution,
    Transform,
    TransformedDistribution,
)

from torch_mist.distributions.caching import add_cache


class DistributionModule(Distribution, nn.Module, ABC):
    def __init__(self, validate_args: bool = False):
        Distribution.__init__(self, validate_args=validate_args)
        nn.Module.__init__(self)

    def __repr__(self):
        return nn.Module.__repr__(self)


class ConditionalDistributionModule(ConditionalDistribution, nn.Module, ABC):
    def __init__(self):
        ConditionalDistribution.__init__(self)
        nn.Module.__init__(self)


class ConditionalTransformedDistributionModule(
    ConditionalTransformedDistribution, nn.Module
):
    def __init__(
        self,
        base_dist: Union[ConditionalDistribution, Distribution],
        transforms: Union[
            ConditionalTransform,
            List[ConditionalTransform],
            Dict[str, Union[ConditionalTransform, Transform]],
            Transform,
            List[Transform],
            None,
        ],
        cached: bool = True,
    ):
        if isinstance(transforms, ConditionalTransform) or isinstance(
            transforms, Transform
        ):
            transforms = [transforms]
        elif isinstance(transforms, dict):
            transforms = [transforms[k] for k in sorted(transforms.keys())]

        if cached:
            transforms = [add_cache(transform) for transform in transforms]

        if transforms is not None:
            transforms = nn.ModuleList(list(transforms))
        else:
            transforms = []

        self._base_dist_repr = base_dist.__repr__()

        nn.Module.__init__(self)
        self.base_dist = base_dist
        self.transforms = transforms

    def condition(self, context):
        base_dist = (
            self.base_dist.condition(context)
            if isinstance(self.base_dist, ConditionalDistribution)
            else self.base_dist
        )
        transforms = [
            (
                t.condition(context)
                if isinstance(t, ConditionalTransform)
                else t
            )
            for t in self.transforms
        ]

        return TransformedDistribution(base_dist, transforms)

    def clear_cache(self):
        pass

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "\n  (base_dist): " + str(self._base_dist_repr).replace(
            "\n", "\n  "
        )
        s += "\n  (transforms): " + str(self.transforms).replace("\n", "\n  ")
        s += "\n"
        return s


class TransformedDistributionModule(DistributionModule):
    def __init__(
        self,
        base_dist: Distribution,
        transforms: Union[
            Transform, List[Transform], Dict[str, Transform], None
        ],
        cached: bool = True,
    ):
        super().__init__()

        self.base_dist = base_dist

        if isinstance(transforms, ConditionalTransform) or isinstance(
            transforms, Transform
        ):
            transforms = [transforms]
        elif hasattr(transforms, "keys"):
            transforms = [transforms[k] for k in sorted(transforms.keys())]

        if cached:
            transforms = [add_cache(transform) for transform in transforms]

        self.transforms = nn.ModuleList(transforms)

    def rsample(self, sample_shape=torch.Size()):
        return TransformedDistribution(
            base_distribution=self.base_dist,
            transforms=[t for t in self.transforms],
        ).rsample(sample_shape)

    def log_prob(self, value):
        return TransformedDistribution(
            base_distribution=self.base_dist,
            transforms=[t for t in self.transforms],
            validate_args=False,
        ).log_prob(value)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "\n  (base_dist): " + str(self.base_dist).replace("\n", "\n  ")
        s += "\n  (transforms): " + str(self.transforms).replace("\n", "\n  ")
        s += "\n)"
        return s
