from typing import Union

from torch.distributions import Transform
from pyro.distributions import (
    TransformModule,
    ConditionalTransformModule,
    ConditionalTransform,
)

from torch_mist.utils.caching import cached_method


class CachedTransformModule(TransformModule):
    def __init__(self, transform: Transform):
        super().__init__()
        self._transform = transform
        self.domain = transform.domain
        self.codomain = transform.codomain
        self.bijective = transform.bijective

    @cached_method
    def _call(self, x):
        return self._transform._call(x)

    @cached_method
    def _inverse(self, y):
        return self._transform._inverse(y)

    @cached_method
    def log_abs_det_jacobian(self, x, y):
        return self._transform.log_abs_det_jacobian(x, y)

    def __repr__(self):
        return "Cached_" + self._transform.__repr__()


class CachedConditionalTransformModule(ConditionalTransformModule):
    def __init__(self, conditional_transform: ConditionalTransform):
        super().__init__()
        self._conditional_transform = conditional_transform

    @cached_method
    def condition(self, context):
        return CachedTransformModule(
            self._conditional_transform.condition(context)
        )

    def __repr__(self):
        return "Cached_" + self._conditional_transform.__repr__()


def add_cache(transform: Union[Transform, ConditionalTransform]):
    if isinstance(transform, Transform):
        return CachedTransformModule(transform)
    elif isinstance(transform, ConditionalTransform):
        return CachedConditionalTransformModule(transform)
    else:
        raise ValueError()
