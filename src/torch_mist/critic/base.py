from typing import Optional

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution
from torch import nn


class Critic(nn.Module):
    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                ) -> torch.Tensor:
        '''
        Compute the value of the critic evaluated at the pair (x, y)
        :param x: a tensor representing x
        :param y: a tensor representing y
        :return: The value of the ratio estimator on the given pair
        '''
        raise NotImplemented()



