from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Optional, List, Union
import numpy as np

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Normal, Distribution


class JointDistribution:
    labels = []

    def entropy(self, labels: Union[str, List[str]]) -> Optional[float]:
        return None

    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


class MultivariateCorrelatedNormalMixture(JointDistribution):
    labels = ["x", "y"]

    def __init__(
            self,
            rho: float = 0.95,
            sigma: float = 0.1,
            epsilon: float = 0.15,
            delta: float = 1.5,
            n_dim: int = 5,
            device: str = "cpu",
            n_samples_estimation: int = 100000
    ):
        self.n_dim = n_dim
        self.n_samples_estimation = n_samples_estimation
        self._h_x = None
        self._h_xy = None

        covariance = torch.FloatTensor([
            [1, rho],
            [rho, 1]
        ]) * sigma

        mu_1 = torch.zeros(n_dim, 1, 2)
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
            covariance[0, 0] ** 0.5
        )

        # Store the marginal distribution for one dimension
        self.p_x = MixtureSameFamily(
            Categorical(probs=torch.zeros(n_dim, 4) + 1.0/4.0),
            self.x_component_dist
        )

        self.component_dist = MultivariateNormal(
            loc.to(device),
            covariance.unsqueeze(0).unsqueeze(1).to(device)
        )

        self.p_xy = MixtureSameFamily(
            Categorical(probs=torch.zeros(n_dim, 4).to(device) + 0.25), self.component_dist
        )

    def _compute_entropies(self):
        xy = self.sample(torch.Size([self.n_samples_estimation]))
        x, y = xy['x'], xy['y']
        samples = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)

        n_components = 4

        h_x = - (
                torch.logsumexp(self.x_component_dist.log_prob(samples[:, :, 0].unsqueeze(-1)), -1).mean()
                - np.log(n_components)
        ) * self.n_dim

        h_xy = - (
                torch.logsumexp(self.component_dist.log_prob(samples.unsqueeze(-2)), -1).mean()
                - np.log(n_components)
        ) * self.n_dim

        self._h_x = h_x.item()
        self._h_xy = h_xy.item()

    def entropy(self, labels: Union[str, List[str]]) -> Optional[float]:
        assert set(labels) <= set(self.labels), f"Labels {labels} not in {self.labels}"
        assert labels in ["x", "y", ["x", "y"]]
        if self._h_x is None:
            self._compute_entropies()
        if labels == "x":
            return self._h_x
        elif labels == "y":
            return self._h_x
        elif labels == ["x", "y"]:
            return self._h_xy
        else:
            return None

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Dict[str, torch.Tensor]:
        xy = self.p_xy.sample(sample_shape)
        x, y = torch.chunk(xy, 2, -1)
        x = x.squeeze(-1)
        y = y.squeeze(-1)

        return {'x': x, 'y': y}

#
# class AttributeResampledDistribution(Distribution, ABC):
#     def __init__(self, base_dist: Distribution, attr_dist: ConditionalDistribution):
#         super(AttributeResampledDistribution, self).__init__(validate_args=False)
#         self.base_dist = base_dist
#         self.attr_dist = attr_dist
#
#     @property
#     def h_a(self) -> float:
#         return np.log(2) * self.base_dist.n_dim
#
#     @property
#     def h_y(self) -> float:
#         return self.base_dist.h_y
#
#     @property
#     def h_x(self) -> float:
#         return self.base_dist.h_x
#
#     @property
#     def mi(self) -> float:
#         return self.base_dist.mi
#
#     @property
#     def h_xy(self) -> float:
#         return self.base_dist.h_xy
#
#     def sample(self, sample_shape: torch.Size = torch.Size([])):
#         d = self.base_dist.sample(sample_shape)
#         x, y0 = d['x'], d['y']
#         n_dim = x.shape[-1]
#
#         # Note here we are considering the same distributions on all the feature dimensions
#         x = x.view(-1)
#         a = self.attr_dist.condition(x).sample()
#
#         y = x * 0
#         sampled = (x * 0).long()
#
#         # Simple rejection sampling
#         while sampled.sum() < x.numel():
#             d = self.base_dist.sample(sample_shape)
#             x_, y_ = d['x'].view(-1), d['y'].view(-1)
#             a_ = self.attr_dist.condition(x_).sample()
#             keep = (a == a_).long() * (1 - sampled)
#             y[keep == 1] += y_[keep == 1]
#             sampled[keep == 1] += 1
#
#         x = x.view(-1, n_dim)
#         y = y.view(-1, n_dim)
#         a = a.view(-1, n_dim)
#
#         if len(sample_shape) == 0:
#             x = x.squeeze(0)
#             y0 = y0.squeeze(0)
#             y = y.squeeze(0)
#             a = a.squeeze(0)
#
#         return {'x': x, 'y': y0, 'y_': y, 'a': a}
#


class SignResampledDistribution(JointDistribution):
    labels = ["x", "y", "y_", "a"]

    def __init__(self, base_dist: MultivariateCorrelatedNormalMixture, n_negatives: int = 1):
        super(SignResampledDistribution, self).__init__()
        self.base_dist = base_dist
        self.n_negatives = n_negatives

    def entropy(self, labels: Union[str, List[str]]) -> Optional[float]:
        assert set(labels) <= set(self.labels), f"Labels {labels} not in {self.labels}"
        assert labels in ["x", "y", "a", ["x", "y"]]
        if labels == "a":
            return np.log(2) * self.base_dist.n_dim
        if labels == "x":
            return self.base_dist.entropy("x")
        elif labels == "y":
            return self.base_dist.entropy("y")
        elif labels == ["x", "y"]:
            return self.base_dist.entropy(["x", "y"])
        else:
            return None

    def compute_attributes(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).long()

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        d = self.base_dist.sample(sample_shape)
        a = self.compute_attributes(d['x'])

        a = a.unsqueeze(1).repeat(1, self.n_negatives, 1)
        d['a'] = a

        # Note here we are considering the same distributions on all the feature dimensions
        if len(sample_shape) == 0:
            n_samples = 1
        else:
            n_samples = sample_shape[0]
            assert len(sample_shape) == 1, "Only 1D sample shapes supported"

        neg_shape = torch.Size([n_samples, self.n_negatives])

        d2 = self.base_dist.sample(neg_shape)
        x_, y_ = d2['x'], d2['y']

        a_ = self.compute_attributes(x_)

        # The mask is 1 if both a are 0 or 1, -1 otherwise
        mask = ((a_ + a) != 1).float() * 2 - 1
        # Flip the y values if a differs
        y_ *= mask

        d['y_'] = y_

        return d
