import torch
from torch import nn

from torch_mist.baseline import Baseline

CRITIC_TYPE = "critic_type"
SEPARABLE_CRITIC = "separable"
JOINT_CRITIC = "joint"
CRITIC_TYPES = [JOINT_CRITIC, SEPARABLE_CRITIC]


class Critic(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the value of the critic evaluated at the pair (x, y)
        :param x: a tensor representing x
        :param y: a tensor representing y
        :return: The value of the ratio estimator on the given pair
        """
        raise NotImplementedError()
