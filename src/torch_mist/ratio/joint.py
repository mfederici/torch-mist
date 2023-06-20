from typing import List

import torch
from pyro.nn import DenseNN
from torch import nn

from src.torch_mist.models.ratio.base import UnnormalizedRatioEstimator


class JointUnnormalizedRatioEstimator(UnnormalizedRatioEstimator):
    def __init__(
            self,
            joint_net: nn.Module,
        ):
        '''
        Model the critic as a function of (x,y).
        :param joint_net: a (learnable) model returning a real number when given pairs (x,y)
        '''

        super(JointUnnormalizedRatioEstimator, self).__init__()
        self.joint_net = joint_net

    def forward(self, x, y) -> torch.Tensor:
        if x.ndim < y.ndim:
            x = x.unsqueeze(1)

        # hack to expand to the same shape without specifying the number of repeats
        x = x + y * 0
        y = y + x * 0

        xy = torch.cat([x, y], -1)

        return self.joint_net(xy).squeeze(-1)


class JointRatioEstimatorMLP(JointUnnormalizedRatioEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
        ):
        '''
        Model the critic as a function of (x,y).
        :param joint_net: a (learnable) model returning a real number when given pairs (x,y)
        '''

        joint_net = DenseNN(input_dim=x_dim + y_dim, hidden_dims=hidden_dims, param_dims=[1])

        super(JointRatioEstimatorMLP, self).__init__(joint_net=joint_net)
