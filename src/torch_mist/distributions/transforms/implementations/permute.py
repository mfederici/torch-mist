import pyro.distributions.transforms as transforms
import torch
from pyro.distributions import TransformModule

from torch import nn


class Permute(transforms.Permute, TransformModule):
    def __init__(self, permutation, *, dim=-1, cache_size=1):
        nn.Module.__init__(self)
        transforms.Permute.__init__(
            self, permutation=permutation, dim=dim, cache_size=cache_size
        )
        self.register_buffer("placeholder", torch.zeros(1))

    def update_device(self):
        if self.permutation.device != self.placeholder.device:
            self.permutation = self.permutation.to(self.placeholder.device)

    def _call(self, x):
        self.update_device()
        return super(Permute, self)._call(x)

    def _inverse(self, y):
        self.update_device()
        return super(Permute, self)._inverse(y)

    def log_abs_det_jacobian(self, x, y):
        self.update_device()
        return super(Permute, self).log_abs_det_jacobian(x, y)
