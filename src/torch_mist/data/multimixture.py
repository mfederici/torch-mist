import torch
from pyro.distributions import Independent
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Normal, Distribution


class MultivariateCorrelatedNormalMixture(MixtureSameFamily):
    def __init__(
            self,
            rho: float = 0.95,
            sigma: float = 0.1,
            epsilon: float = 0.15,
            delta: float = 1.5,
            n_dim: int = 5
    ):
        covariance = torch.eye(n_dim * 2)
        for dim in range(n_dim):
            covariance[dim, n_dim + dim] = covariance[n_dim + dim, dim] = rho

        covariance = covariance * sigma

        mu = torch.zeros(4, 2, n_dim)

        mu[0, 0] = epsilon + delta
        mu[0, 1] = -epsilon + delta
        mu[1, 0] = -epsilon - delta
        mu[1, 1] = epsilon - delta
        mu[2, 0] = epsilon - delta
        mu[2, 1] = -epsilon - delta
        mu[3, 0] = -epsilon + delta
        mu[3, 1] = epsilon + delta

        self.x_component_dist = Normal(
            mu[:, 0].T,
            sigma**0.5
        )

        # Store the marginal distribution for one dimension
        self.p_x = Independent(MixtureSameFamily(
            Categorical(
                logits=torch.zeros(n_dim, 4)
            ),
            self.x_component_dist
        ), 1)

        self.component_dist = MultivariateNormal(
            mu.view(4, -1),
            covariance.unsqueeze(0)
        )

        super().__init__(
            Categorical(logits=torch.zeros(4)), self.component_dist
        )
