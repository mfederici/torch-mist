from typing import Dict

import torch
from pyro.distributions import Independent
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Normal, Distribution

from torch_mist.distributions.joint import JointDistribution


class MultivariateCorrelatedNormalMixture(JointDistribution):
    def __init__(
            self,
            rho: float = 0.95,
            sigma: float = 0.1,
            epsilon: float = 0.15,
            delta: float = 1.5,
            n_dim: int = 5,
            device: torch.device = torch.device('cpu')
    ):
        covariance = torch.eye(2).to(device)
        covariance[0, 1] = covariance[1, 0] = rho
        covariance = covariance * sigma

        mu = torch.zeros(n_dim, 4, 2).to(device)

        mu[:, 0, 0] = epsilon + delta
        mu[:, 0, 1] = -epsilon + delta
        mu[:, 1, 0] = -epsilon - delta
        mu[:, 1, 1] = epsilon - delta
        mu[:, 2, 0] = epsilon - delta
        mu[:, 2, 1] = -epsilon - delta
        mu[:, 3, 0] = -epsilon + delta
        mu[:, 3, 1] = epsilon + delta

        self.x_component_dist = Normal(
            mu[:, :, 0],
            sigma**0.5
        )

        # Store the marginal distribution for one dimension
        self.p_X = Independent(
            MixtureSameFamily(
                Categorical(
                    logits=torch.zeros(n_dim, 4).to(device)
                ),
                self.x_component_dist),
            1
        )

        self.component_dist = MultivariateNormal(
            mu,
            covariance.unsqueeze(0).unsqueeze(1)
        )

        p_XY = Independent(
            MixtureSameFamily(
                Categorical(
                    logits=torch.zeros(n_dim, 4).to(device)
                ),
                self.component_dist
            ), 1
        )

        super().__init__(joint_dist=p_XY, dims=[1, 1], labels=['x', 'y'], squeeze=True)


