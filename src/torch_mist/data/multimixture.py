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
        covariance = torch.FloatTensor([
            [1, rho],
            [rho, 1]
        ])*sigma

        mu_1 = torch.zeros(n_dim, 1, 2)
        # offset1 = torch.Tensor([-epsilon, epsilon])
        # offset2 = torch.Tensor([-delta, delta])

        mu_1[:, :, 0] = epsilon + delta
        mu_1[:, :, 1] = -epsilon + delta

        mu_2 = torch.zeros(n_dim, 1, 2)
        mu_2[:, :, 0] = -epsilon - delta
        mu_2[:, :, 1] = epsilon - delta

        mu_3 = torch.zeros(n_dim, 1, 2)
        mu_3[:, :, 0] = epsilon - delta
        mu_3[:, :, 1] = -epsilon - delta

        mu_4 = torch.zeros(n_dim, 1, 2)
        mu_4[:, :, 0] = -epsilon + delta
        mu_4[:, :, 1] = epsilon + delta

        loc = torch.cat([mu_1, mu_2, mu_3, mu_4], -2)

        self.x_component_dist = Normal(
            loc[:, :, 0],
            covariance[0, 0]**0.5
        )

        # Store the marginal distribution for one dimension
        self.p_x = Independent(MixtureSameFamily(
            Categorical(probs=torch.zeros(n_dim, 4) + 0.25),
            self.x_component_dist
        ), 1)

        self.component_dist = MultivariateNormal(
            loc,
            covariance.unsqueeze(0).unsqueeze(1)
        )

        super().__init__(
            Categorical(probs=torch.zeros(n_dim, 4) + 0.25), self.component_dist
        )
