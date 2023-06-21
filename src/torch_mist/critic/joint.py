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
            x = x.unsqueeze(1)

        # hack to expand to the same shape without specifying the number of repeats
        x = x + y * 0
        y = y + x * 0

        xy = torch.cat([x, y], -1)

        return self.joint_net(xy).squeeze(-1)
