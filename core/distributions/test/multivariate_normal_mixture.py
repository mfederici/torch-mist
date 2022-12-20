from typing import Callable

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Normal, Distribution


class MultivariateCorrelatedNormalMixture(MixtureSameFamily):
    def __init__(self, rho: float = 0.95, epsilon: float = 0.3, n_dim: int = 5, device: str = "cpu"):
        covariance = torch.FloatTensor([
            [1, rho],
            [rho, 1]
        ])

        mu_1 = torch.zeros(n_dim, 1, 2)
        mu_1[:, :, 0] = epsilon
        mu_1[:, :, 1] = -epsilon

        mu_2 = torch.zeros(n_dim, 1, 2)
        mu_2[:, :, 0] = -epsilon
        mu_2[:, :, 1] = epsilon

        loc = torch.cat([mu_1, mu_2], -2)

        self.x_component_dist = Normal(
            loc[:, 0],
            1
        )

        # Store the marginal distribution for one dimension
        self.p_x = MixtureSameFamily(
            Categorical(probs=torch.zeros(n_dim, 2) + 0.5),
            self.x_component_dist
        )
        self.component_dist = MultivariateNormal(
            loc.to(device),
            covariance.unsqueeze(0).unsqueeze(1).to(device)
        )

        super().__init__(
            Categorical(probs=torch.zeros(n_dim, 2).to(device) + 0.5), self.component_dist
        )

    def sample(self, sample_shape=torch.Size()):
        xy = super().sample(sample_shape)
        x, y = torch.chunk(xy, 2, -1)
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        return x, y

class AttributeResampledDistribution(Distribution):
    def __init__(self, base_dist: Distribution, attr_dist: ConditionalDistribution):
        super(AttributeResampledDistribution, self).__init__(validate_args=False)
        self.base_dist = base_dist
        self.attr_dist = attr_dist

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        x, y0 = self.base_dist.sample(sample_shape)
        n_dim = x.shape[-1]

        # Note here we are considering the same distributions on all the feature dimensions
        x = x.view(-1)
        a = self.attr_dist.condition(x).sample()

        y = x * 0
        sampled = (x * 0).long()

        # Simple rejection sampling
        while sampled.sum() < x.numel():
            x_, y_ = self.base_dist.sample(sample_shape)
            x_ = x_.view(-1)
            y_ = y_.view(-1)
            a_ = self.attr_dist.condition(x_).sample()
            keep = (a == a_).long() * (1 - sampled)
            y[keep == 1] += y_[keep == 1]
            sampled[keep == 1] += 1

        x = x.view(-1, n_dim)
        y = y.view(-1, n_dim)
        a = a.view(-1, n_dim)

        return x, y0, y, a


class SignResampledDistribution(Distribution):
    def __init__(self, base_dist: Distribution):
        super(SignResampledDistribution, self).__init__(validate_args=False)
        self.base_dist = base_dist

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        x, y0 = self.base_dist.sample(sample_shape)
        n_dim = x.shape[-1]
        y0 = y0.squeeze(-1)

        # Note here we are considering the same distributions on all the feature dimensions
        x = x.view(-1)
        a = (x > 0).long()

        x_, y = self.base_dist.sample(sample_shape)
        x_ = x_.view(-1)
        y = y.view(-1)

        a_ = (x_ > 0).long()
        mask = (a_+a) == 1
        y[mask] = -y[mask]

        x = x.view(-1, n_dim)
        y = y.view(-1, n_dim)
        y0 = y0.view(-1, n_dim)
        a = a.view(-1, n_dim)

        return x, y0, y, a

