from typing import Dict, List

import torch
from torch.distributions import (
    MultivariateNormal,
    Normal,
    Independent,
    Distribution,
)

from torch_mist.distributions.joint.base import JointDistribution
from torch_mist.distributions.joint.wrapper import TorchJointDistribution


class JointMultivariateNormal(TorchJointDistribution):
    def __init__(
        self,
        n_dim: int,
        rho: float = 0.9,
        sigma: float = 1,
        device: torch.device = torch.device("cpu"),
    ):
        loc = torch.zeros(n_dim, 2).to(device)
        covariance = torch.eye(2).to(device)
        covariance[0, 1] = covariance[1, 0] = rho
        covariance = covariance * sigma
        covariance = covariance.unsqueeze(0).repeat(n_dim, 1, 1)

        p_XY = Independent(MultivariateNormal(loc, covariance), 1)

        self.p_X = Independent(Normal(torch.zeros(n_dim).to(device), sigma), 1)

        super().__init__(torch_dist=p_XY, variables=["x", "y"])

    def _marginal(self, variables: List[str]) -> JointDistribution:
        assert len(variables) == 1
        variable = variables[0]

        return TorchJointDistribution(
            torch_dist=self.p_X,
            variables=[variable],
        )

    def _entropy(self, variables: List[str]) -> torch.Tensor:
        if len(variables) == 2:
            return self.torch_dist.entropy()
        else:
            return self.p_X.entropy()
