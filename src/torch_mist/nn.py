from typing import List, Callable, Optional, Tuple

import torch
import torch.nn as nn
from pyro.nn import DenseNN

from torch_mist.utils.caching import cached_method


class Normalize(nn.Module):
    def forward(self, x):
        return x / (x**2).sum(-1).unsqueeze(-1) ** 0.5


class CachedModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    @cached_method
    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)

    def __repr__(self):
        return "cached_" + self.module.__repr__()


def dense_nn(
    input_dim: int,
    output_dim: int,
    hidden_dims: Optional[List[int]] = None,
    nonlinearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> nn.Module:
    if nonlinearity is None:
        nonlinearity = nn.ReLU(True)

    if hidden_dims is None:
        hidden_dims = []
    if len(hidden_dims) == 0:
        return nn.Linear(input_dim, output_dim)
    else:
        return DenseNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            param_dims=[output_dim],
            nonlinearity=nonlinearity,
        )


def multi_head_dense_nn(
    input_dim: int,
    output_dim: int,
    n_shared_layers: int,
    hidden_dims: List[int],
    n_heads: int,
    cached_shared_forward: bool = True,
    nonlinearity: nn.Module = nn.ReLU(True),
) -> Tuple[nn.Module, ...]:
    hidden_dims = [input_dim] + hidden_dims

    if n_shared_layers < 0:
        n_shared_layers = len(hidden_dims) + n_shared_layers

    assert 0 <= n_shared_layers <= len(hidden_dims)

    if n_shared_layers == 0:
        shared_net = None
    else:
        shared_net = dense_nn(
            input_dim=input_dim,
            output_dim=hidden_dims[n_shared_layers],
            hidden_dims=hidden_dims[1:n_shared_layers],
            nonlinearity=nonlinearity,
        )
        shared_net = nn.Sequential(
            shared_net,
            nonlinearity,
        )

        if cached_shared_forward:
            shared_net = CachedModule(shared_net)

    nets = []
    for _ in range(n_heads):
        head = dense_nn(
            input_dim=hidden_dims[n_shared_layers],
            output_dim=output_dim,
            hidden_dims=hidden_dims[n_shared_layers + 1 :],
            nonlinearity=nonlinearity,
        )
        if n_shared_layers == 0:
            nets.append(head)
        elif n_shared_layers == len(hidden_dims):
            nets.append(shared_net)
        else:
            nets.append(nn.Sequential(shared_net, head))

    return tuple(nets)
