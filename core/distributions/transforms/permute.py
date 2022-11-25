import pyro.distributions.transforms as transforms
import torch
from pyro.distributions import TransformModule

from torch import nn


class Permute(transforms.Permute, TransformModule):
    def __init__(self, permutation, *, dim=- 1, cache_size=1):
        nn.Module.__init__(self)
        transforms.Permute.__init__(self, permutation=permutation,  dim=dim, cache_size=cache_size)
        self.register_buffer('placeholder', torch.zeros(1))

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


def permute(input_dim, permutation=None, dim=-1):
    """
    A helper function to create a :class:`~pyro.distributions.transforms.Permute`
    object for consistency with other helpers.

    :param input_dim: Dimension(s) of input variable to permute. Note that when
        `dim < -1` this must be a tuple corresponding to the event shape.
    :type input_dim: int
    :param permutation: Torch tensor of integer indices representing permutation.
        Defaults to a random permutation.
    :type permutation: torch.LongTensor
    :param dim: the tensor dimension to permute. This value must be negative and
        defines the event dim as `abs(dim)`.
    :type dim: int

    """
    if dim < -1 or not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError(
                "event shape {} must have same length as event_dim {}".format(
                    input_dim, -dim
                )
            )
        input_dim = input_dim[dim]

    if permutation is None:
        permutation = torch.randperm(input_dim)
    return Permute(permutation, dim=dim)
