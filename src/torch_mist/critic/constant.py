import torch

from .base import Critic
from torch_mist.utils.shape import expand_to_same_shape


class ConstantCritic(Critic):
    def __init__(self, value: float = 0):
        super().__init__()
        self.value = value

    def forward(self, x, y) -> torch.Tensor:
        x, y = expand_to_same_shape(x, y)
        return x[..., 0].detach() * 0 + self.value
