import numpy as np
import torch
from pyro.distributions import TransformModule, Independent
from torch.distributions import constraints, Normal


class SplitTransform(TransformModule):
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, input_dim, out_dim):
        super().__init__(cache_size=1)
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.register_buffer('mu', torch.zeros(np.abs(out_dim-input_dim)))

    def dist(self):
        return Independent(Normal(self.mu, 1), 1)

    def add_dims(self, x):
        x_right = self.dist().sample(x.shape[0]).sample()
        y = torch.cat([x, x_right], -1)
        return y, x_right

    def remove_dims(self, y):
        x, x_right = torch.split(y, [
            min(self.input_dim, self.out_dim),
            np.abs(self.input_dim - self.dim_to_add)
        ], -1)
        return x, x_right

    def _call(self, x):
        if self.out_dim > self.input_dim:
            y, x_right = self.add_dims(x)
        else:
            y, x_right = self.remove_dims(x)
        self._cached_x_right = x_right
        return y

    def _inverse(self, y):
        if self.out_dim > self.input_dim:
            x, x_right = self.remove_dims(y)
        else:
            x, x_right = self.add_dims(y)
        return x

    def log_abs_det_jacobian(self, x, y):
        x_old, y_old = self._cached_x_y
        if not(x is x_old and y is y_old):
            raise NotImplementedError()
        x_right = self._cached_x_right

        return self.dist().log_prob(x_right)