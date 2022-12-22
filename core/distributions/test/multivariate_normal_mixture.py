from abc import ABC
from typing import Dict
import numpy as np


import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Normal, Distribution


class MultivariateCorrelatedNormalMixture(MixtureSameFamily):
    def __init__(self, rho: float = 0.95, sigma: float = 0.1, epsilon: float = 0.15,  delta: float = 1.5, n_dim: int = 5, device: str = "cpu"):
        covariance = torch.FloatTensor([
            [1, rho],
            [rho, 1]
        ])*sigma

        self.n_dim = n_dim

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
            covariance[0,0]**0.5
        )

        # Store the marginal distribution for one dimension
        self.p_x = MixtureSameFamily(
            Categorical(probs=torch.zeros(n_dim, 4) + 0.25),
            self.x_component_dist
        )
        self.component_dist = MultivariateNormal(
            loc.to(device),
            covariance.unsqueeze(0).unsqueeze(1).to(device)
        )

        super().__init__(
            Categorical(probs=torch.zeros(n_dim, 4).to(device) + 0.25), self.component_dist
        )

    @property
    def h_y(self) -> float:
        n_samples = 100000
        xy = self.sample([n_samples])
        x, y = xy['x'], xy['y']
        samples = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)

        n_components = 4
        h_xy = (-torch.logsumexp(self.component_dist.log_prob(samples.unsqueeze(-2)), -1).mean() + np.log(
            n_components)) * self.n_dim
        h_x = h_y = (-torch.logsumexp(self.x_component_dist.log_prob(samples[:, :, 0].unsqueeze(-1)),
                                      -1).mean() + np.log(n_components)) * self.n_dim

        return h_y.item()

    @property
    def h_x(self) -> float:
        return self.h_y

    @property
    def h_xy(self) -> float:
        n_samples = 100000
        xy = self.sample([n_samples])
        x, y = xy['x'], xy['y']
        samples = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)

        n_components = 4
        h_xy = (-torch.logsumexp(self.component_dist.log_prob(samples.unsqueeze(-2)), -1).mean() + np.log(
            n_components)) * self.n_dim

        return h_xy.item()

    @property
    def mi(self) -> float:
        return self.h_x + self.h_y - self.h_xy

    def sample(self, sample_shape=torch.Size()) -> Dict[str, torch.Tensor]:
        xy = super().sample(sample_shape)
        x, y = torch.chunk(xy, 2, -1)
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        return {'x': x, 'y': y}


class AttributeResampledDistribution(Distribution, ABC):
    def __init__(self, base_dist: Distribution, attr_dist: ConditionalDistribution):
        super(AttributeResampledDistribution, self).__init__(validate_args=False)
        self.base_dist = base_dist
        self.attr_dist = attr_dist

    @property
    def h_a(self) -> float:
        return np.log(2) * self.base_dist.n_dim

    @property
    def h_y(self) -> float:
        return self.base_dist.h_y

    @property
    def h_x(self) -> float:
        return self.base_dist.h_x

    @property
    def mi(self) -> float:
        return self.base_dist.mi

    @property
    def h_xy(self) -> float:
        return self.base_dist.h_xy

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        d = self.base_dist.sample(sample_shape)
        x, y0 = d['x'], d['y']
        n_dim = x.shape[-1]

        # Note here we are considering the same distributions on all the feature dimensions
        x = x.view(-1)
        a = self.attr_dist.condition(x).sample()

        y = x * 0
        sampled = (x * 0).long()

        # Simple rejection sampling
        while sampled.sum() < x.numel():
            d = self.base_dist.sample(sample_shape)
            x_, y_ = d['x'].view(-1), d['y'].view(-1)
            a_ = self.attr_dist.condition(x_).sample()
            keep = (a == a_).long() * (1 - sampled)
            y[keep == 1] += y_[keep == 1]
            sampled[keep == 1] += 1

        x = x.view(-1, n_dim)
        y = y.view(-1, n_dim)
        a = a.view(-1, n_dim)

        if len(sample_shape) == 0:
            x = x.squeeze(0)
            y0 = y0.squeeze(0)
            y = y.squeeze(0)
            a = a.squeeze(0)

        return {'x': x, 'y': y0, 'y_': y, 'a': a}


class SignResampledDistribution(Distribution, ABC):
    def __init__(self, base_dist: Distribution):
        super(SignResampledDistribution, self).__init__(validate_args=False)
        self.base_dist = base_dist

    @property
    def h_a(self) -> float:
        return np.log(2) * self.base_dist.n_dim

    @property
    def h_y(self) -> float:
        return self.base_dist.h_y

    @property
    def h_x(self) -> float:
        return self.base_dist.h_x

    @property
    def mi(self) -> float:
        return self.base_dist.mi

    @property
    def h_xy(self) -> float:
        return self.base_dist.h_xy

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        d = self.base_dist.sample(sample_shape)
        x, y0 = d['x'], d['y']

        n_dim = x.shape[-1]
        y0 = y0.squeeze(-1)

        # Note here we are considering the same distributions on all the feature dimensions
        x = x.view(-1)
        a = (x > 0).long()

        d = self.base_dist.sample(sample_shape)
        x_, y = d['x'].view(-1), d['y'].view(-1)

        a_ = (x_ > 0).long()
        mask = (a_+a) == 1
        y[mask] = -y[mask]

        x = x.view(-1, n_dim)
        y = y.view(-1, n_dim)
        y0 = y0.view(-1, n_dim)
        a = a.view(-1, n_dim)

        if len(sample_shape) == 0:
            x = x.squeeze(0)
            y0 = y0.squeeze(0)
            y = y.squeeze(0)
            a = a.squeeze(0)

        return {'x': x, 'y': y0, 'y_': y, 'a': a}

