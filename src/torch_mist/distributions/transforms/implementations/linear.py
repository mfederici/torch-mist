from functools import partial

import numpy as np
import torch
from pyro.distributions import (
    constraints,
    TransformModule,
    ConditionalTransformModule,
)
from pyro.nn import DenseNN
from torch import nn
from torch.distributions import Transform


class ConditionedLinear(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, params, epsilon=1e-6):
        super(ConditionedLinear, self).__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None
        self.epsilon = epsilon

    def _call(self, x):
        loc, log_scale = (
            self._params() if callable(self._params) else self._params
        )
        if not (log_scale.shape == loc.shape):
            log_scale = loc * 0 + log_scale

        self._cached_logDetJ = log_scale.sum(-1)
        y = x * (log_scale.exp() + self.epsilon) + loc
        return y

    def _inverse(self, y):
        loc, log_scale = (
            self._params() if callable(self._params) else self._params
        )
        if not (log_scale.shape == loc.shape):
            log_scale = loc * 0 + log_scale

        x = (y - loc) / (log_scale.exp() + self.epsilon)
        self._cached_logDetJ = log_scale.sum(-1)
        return x

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_logDetJ


class Linear(ConditionedLinear, TransformModule):
    def __init__(
        self, input_dim, loc=None, scale=None, initial_scale=None, epsilon=1e-6
    ):
        super(Linear, self).__init__(self._params)
        if loc is None:
            self.loc = nn.Parameter(torch.zeros(input_dim))
        else:
            self.register_buffer("loc", torch.zeros(input_dim) + loc)
        if initial_scale is None:
            initial_scale = 1.0
        if scale is None:
            self.log_scale = nn.Parameter(
                torch.zeros(input_dim) + np.log(initial_scale)
            )
        else:
            self.register_buffer("log_scale", torch.Tensor([scale]).log())
        self.epsilon = epsilon

    def _params(self):
        return self.loc, self.log_scale


class ConditionalLinear(ConditionalTransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        net,
        loc=None,
        scale=None,
        initial_scale=None,
        epsilon=1e-6,
        skip_connection=False,
    ):
        super(ConditionalLinear, self).__init__()

        self.nn = net
        self.epsilon = epsilon
        self.initial_scale = initial_scale
        self.skip_connection = skip_connection

        if loc is None:
            self.loc = None
        else:
            self.register_buffer("loc", torch.Tensor([loc]).log())

        if initial_scale is None:
            initial_scale = 1.0

        if scale is None:
            self.log_scale = None
            self.initial_scale = initial_scale
        else:
            self.register_buffer("log_scale", torch.Tensor([scale]).log())

    def _params(self, context):
        if self.log_scale is None and self.loc is None:
            loc, log_scale = self.nn(context)
            log_scale = log_scale + np.log(self.initial_scale)
        elif self.loc is None:
            loc = self.nn(context)
            log_scale = self.log_scale
        elif self.log_scale is None:
            loc = self.loc
            log_scale = self.nn(context)
            log_scale = log_scale + np.log(self.initial_scale)
        else:
            loc = self.loc
            log_scale = self.log_scale

        if self.skip_connection and context.shape == loc.shape:
            loc += context

        return loc, log_scale

    def condition(self, context):
        params = partial(self._params, context)
        return ConditionedLinear(params, epsilon=self.epsilon)
