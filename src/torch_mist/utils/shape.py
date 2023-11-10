from typing import Optional, Tuple

import torch


def expand_to_same_shape(*tensors) -> Tuple[torch.Tensor, ...]:
    if len(tensors) == 1:
        assert isinstance(tensors[0], torch.Tensor)
        return tensors[0]
    ndim = tensors[0].ndim

    for tensor in tensors:
        assert tensor.ndim == ndim

    # Find the maximum shape
    max_shape = [
        max([tensor.shape[i] for tensor in tensors]) for i in range(ndim - 1)
    ]
    max_shape += [-1]

    # Expand x and y to the maximum shape
    return tuple([tensor.expand(max_shape) for tensor in tensors])
