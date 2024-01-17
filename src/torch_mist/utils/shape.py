from typing import Tuple

import torch


def expand_to_same_shape(*tensors) -> Tuple[torch.Tensor, ...]:
    if len(tensors) == 1:
        assert isinstance(tensors[0], torch.Tensor)
        return tensors[0]
    ndim = tensors[0].ndim

    # Tensors of integer are considered to have an extra (empty) dimension
    if torch.is_floating_point(tensors[0]):
        ndim -= 1

    for tensor in tensors:
        assert (
            tensor.ndim - (1 if torch.is_floating_point(tensor) else 0) == ndim
        )

    # Find the maximum shape
    max_shape = [
        max([tensor.shape[i] for tensor in tensors]) for i in range(ndim)
    ]

    # Expand x and y to the maximum shape
    expanded_tensors = []
    for tensor in tensors:
        expanded_tensors.append(
            tensor.expand(
                max_shape + ([-1] if torch.is_floating_point(tensor) else [])
            )
        )

    return tuple(expanded_tensors)
