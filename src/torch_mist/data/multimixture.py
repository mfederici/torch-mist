from typing import List, Dict

import numpy as np
import torch
from pyro.distributions import Independent
from torch.distributions import (
    MultivariateNormal,
    MixtureSameFamily,
    Categorical,
    Normal,
)

from torch_mist.distributions.joint.base import JointDistribution
from torch_mist.distributions.joint.wrapper import TorchJointDistribution


class MultivariateCorrelatedNormalMixture(TorchJointDistribution):
    MC_SAMPLES = 100000
    MC_ITERATIONS = 100

    def __init__(
        self,
        rho: float = 0.95,
        sigma: float = 0.1,
        epsilon: float = 0.15,
        delta: float = 1.5,
        n_dim: int = 5,
    ):
        self.n_dim = n_dim
        self.rho = rho
        self.sigma = sigma
        self.epsilon = epsilon
        self.delta = delta

        covariance = torch.eye(2)
        covariance[0, 1] = covariance[1, 0] = rho
        covariance = covariance * sigma

        mu = torch.zeros(n_dim, 4, 2)

        mu[:, 0, 0] = epsilon + delta
        mu[:, 0, 1] = -epsilon + delta
        mu[:, 1, 0] = -epsilon - delta
        mu[:, 1, 1] = epsilon - delta
        mu[:, 2, 0] = epsilon - delta
        mu[:, 2, 1] = -epsilon - delta
        mu[:, 3, 0] = -epsilon + delta
        mu[:, 3, 1] = epsilon + delta

        self.x_component_dist = Normal(mu[:, :, 0], sigma**0.5)

        # Store the marginal distribution for one dimension
        self.p_X = Independent(
            MixtureSameFamily(
                Categorical(logits=torch.zeros(n_dim, 4)),
                self.x_component_dist,
            ),
            1,
        )

        self.component_dist = MultivariateNormal(
            mu, covariance.unsqueeze(0).unsqueeze(1)
        )

        p_XY = Independent(
            MixtureSameFamily(
                Categorical(logits=torch.zeros(n_dim, 4)),
                self.component_dist,
            ),
            1,
        )

        self._cached_estimates = {}

        super().__init__(torch_dist=p_XY, variables=["x", "y"])

    def _marginal(self, variables: List[str]) -> JointDistribution:
        assert len(variables) == 1
        variable = variables[0]

        return TorchJointDistribution(
            torch_dist=self.p_X,
            variables=[variable],
        )

    def _approximate_estimates(self):
        if self.n_dim > 1:
            p_XY = MultivariateCorrelatedNormalMixture(
                n_dim=1,
                rho=self.rho,
                delta=self.delta,
                sigma=self.sigma,
                epsilon=self.epsilon,
            )
        else:
            p_XY = self

        estimates = {
            "H(x)": [],
            "H(y)": [],
            "H(xy)": [],
            "I(x;y)": [],
        }

        p_X = p_XY.marginal("x")
        p_Y = p_XY.marginal("y")

        for i in range(self.MC_ITERATIONS):
            # Sample from the joint
            samples = p_XY.sample([self.MC_SAMPLES])

            # Compute the log-probability of each sample pair
            log_p_xy = p_XY.log_prob(**samples)

            # And the marginal log-probability
            log_p_x = p_X.log_prob(samples["x"])
            log_p_y = p_Y.log_prob(samples["y"])

            # MC estimation of the joint and marginal entropies
            entropy_xy = -torch.mean(log_p_xy)
            entropy_x = -torch.mean(log_p_x)
            entropy_y = -torch.mean(log_p_y)

            estimates["H(x)"].append(entropy_x)
            estimates["H(x)"].append(entropy_y)
            estimates["H(xy)"].append(entropy_xy)
            estimates["I(x;y)"].append(entropy_x + entropy_y - entropy_xy)

        for name, values in estimates.items():
            self._cached_estimates[name] = np.mean(values) * self.n_dim

    def _entropy(self, variables: List[str]) -> torch.Tensor:
        if len(self._cached_estimates) == 0:
            self._approximate_estimates()
        if len(variables) == 2:
            return self._cached_estimates["H(xy)"]

        return self._cached_estimates["H(x)"]

    def _mutual_information(
        self, variable_1: str, variable_2: str
    ) -> torch.Tensor:
        if len(self._cached_estimates) == 0:
            self._approximate_estimates()
        return self._cached_estimates["I(x;y)"]
