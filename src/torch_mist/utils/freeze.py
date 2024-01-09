from typing import Any

from torch import nn


def freeze(function: Any) -> Any:
    if isinstance(function, nn.Module):
        for param in function.parameters():
            param.requires_grad = False

    return function


def n_trainable_parameters(function: Any) -> int:
    if isinstance(function, nn.Module):
        n_parameters = 0
        for param in function.parameters():
            if param.requires_grad:
                n_parameters += param.numel()

        return n_parameters
    else:
        return 0


def is_trainable(function: Any) -> bool:
    if isinstance(function, nn.Module):
        return n_trainable_parameters(function) > 0
    else:
        return False
