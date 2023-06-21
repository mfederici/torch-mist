from typing import Optional, List

import torch
from pyro.nn import DenseNN

from torch import nn

from .base import Critic


class SeparableCritic(Critic):
    def __init__(
            self,
            f_x: Optional[nn.Module] = None,
            f_y: Optional[nn.Module] = None,
            temperature: float = 1.0
    ):
        '''
        Model the critic as the product of two feature extractors
        f(x,y) = f_x(x)^T f_y(y)
        Note that the baseline can return a constant value (e.g. NWJ estimator)
        :param f_x: a (learnable) model returning a real vector of size D when given x. The identity is used if None is specified.
        :param f_y: a (learnable) model returning a real vector of size D when given y. The identity is used if None is specified.
        '''

        super().__init__()
        self.f_x = (lambda x: x) if f_x is None else f_x
        self.f_y = (lambda y: y) if f_y is None else f_y
        self.temperature = temperature

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        f_x = self.f_x(x)
        f_y = self.f_y(y)

        if f_x.ndim < f_y.ndim:
            f_x = f_x.unsqueeze(1)

        # hack to expand to the same shape without specifying the number of repeats
        f_x = f_x + f_y * 0
        f_y = f_y + f_x * 0

        K = torch.einsum('acb, acb -> ac', f_x, f_y)
        return K / self.temperature
