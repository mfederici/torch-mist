from typing import List

import torch
from pyro.nn import DenseNN
from torch import nn

from .base import Critic
from torch_mist.utils.shape import expand_to_same_shape


class JointCritic(Critic):
    def __init__(
        self,
        joint_net: nn.Module,
    ):
        """
        Model the critic as a function of (x,y).
        :param joint_net: a (learnable) model returning a real number when given pairs (x,y)
        """

        super(JointCritic, self).__init__()
        self.joint_net = joint_net

    def forward(self, x, y) -> torch.Tensor:
        x, y = expand_to_same_shape(x, y)

        # Concatenate x and y in order
        xy = torch.cat([x, y], -1)

        return self.joint_net(xy).squeeze(-1)
