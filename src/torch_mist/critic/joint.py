from typing import List

import torch
from pyro.nn import DenseNN
from torch import nn

from .base import Critic


class JointCritic(Critic):
    def __init__(
            self,
            joint_net: nn.Module,
        ):
        '''
        Model the critic as a function of (x,y).
        :param joint_net: a (learnable) model returning a real number when given pairs (x,y)
        '''

        super(JointCritic, self).__init__()
        self.joint_net = joint_net

    def forward(self, x, y) -> torch.Tensor:
        if x.ndim < y.ndim:
            x = x.unsqueeze(0)

        assert x.ndim == y.ndim, f'x.ndim={x.ndim}, y.ndim={y.ndim}'
        n_dims = x.ndim

        # Find the maximum shape
        max_shape = [max(x.shape[i], y.shape[i]) for i in range(n_dims - 1)]
        # Expand x and y to the maximum shape
        x = x.expand(max_shape + [-1])
        y = y.expand(max_shape + [-1])

        # Concatenate x and y in order
        xy = torch.cat([x, y], -1)

        return self.joint_net(xy).squeeze(-1)
