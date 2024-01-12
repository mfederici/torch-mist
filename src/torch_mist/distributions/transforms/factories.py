from typing import Optional, List

import torch
from torch import nn
from pyro.nn import DenseNN

from torch_mist.distributions.transforms.implementations.permute import Permute
from torch_mist.distributions.transforms.implementations.linear import (
    Linear,
    ConditionalLinear,
)
from torch_mist.distributions.transforms.implementations.normalize import (
    EMANormalize,
)


def linear(
    input_dim: int,
    loc: Optional[float] = None,
    scale: Optional[float] = None,
    initial_scale: Optional[float] = None,
) -> Linear:
    return Linear(input_dim, scale=scale, initial_scale=initial_scale, loc=loc)


def conditional_linear(
    input_dim: int,
    context_dim: int,
    hidden_dims: Optional[List[int]] = None,
    scale: Optional[float] = None,
    initial_scale: Optional[float] = None,
    nonlinearity: nn.Module = nn.ReLU(True),
) -> ConditionalLinear:
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    if scale is None:
        nn = DenseNN(
            input_dim=context_dim,
            hidden_dims=hidden_dims,
            param_dims=[input_dim, input_dim],
            nonlinearity=nonlinearity,
        )
    else:
        nn = DenseNN(
            context_dim,
            hidden_dims,
            param_dims=[input_dim],
            nonlinearity=nonlinearity,
        )
    return ConditionalLinear(nn, scale=scale, initial_scale=initial_scale)


def permute(
    input_dim: int, permutation: Optional[List[int]] = None, dim: int = -1
) -> Permute:
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


def emanorm(
    input_dim: int, gamma: float = 0.99, normalize_inverse: bool = True
) -> EMANormalize:
    return EMANormalize(
        input_dim=input_dim, gamma=gamma, normalize_inverse=normalize_inverse
    )
