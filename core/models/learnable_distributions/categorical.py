from typing import List

import torch
from torch import nn
from torch.distributions import Independent, Categorical, Distribution
from pyro.nn import DenseNN
from pyro.distributions import ConditionalDistribution

from core.distributions.conditional import ConditionalCategorical


class ConditionalCategoricalMLP(ConditionalCategorical):
    def __init__(
            self,
            y_dim: int,
            n_classes: int,
            hidden_dims: List[int],
            a_dim: int = 1,
    ):

        net = DenseNN(input_dim=y_dim, hidden_dims=hidden_dims, param_dims=[n_classes] * a_dim)
        super().__init__(net)


class ConditionalCategoricalLinear(ConditionalDistribution, nn.Module):
    def __init__(
            self,
            y_dim: int,
            n_classes: int,
            a_dim: int = 1,
    ):
        super(ConditionalCategoricalLinear, self).__init__()
        self.w = nn.Linear(y_dim, n_classes * a_dim)
        self.a_dim = a_dim
        self.n_classes = n_classes

    def condition(self, x):
        shape = list(x.shape[:-1]) + [self.a_dim, self.n_classes]
        logits = self.w(x).view(*shape)
        return Independent(Categorical(logits=logits), 1)


class LearnableCategorical(Distribution, nn.Module):
    def __init__(
            self,
            n_classes: int,
            a_dim: int = 1,
    ):
        nn.Module.__init__(self)
        Distribution.__init__(self, validate_args=False)
        self.a_dim = a_dim
        self.n_classes = n_classes
        self.logits = nn.Parameter(torch.zeros(a_dim, n_classes))

    def log_prob(self, value):
        return Independent(Categorical(logits=self.logits), 1).log_prob(value)

    def sample(self, sample_shape=torch.Size()):
        return Categorical(logits=self.logits).sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return Categorical(logits=self.logits).rsample(sample_shape)

    def __repr__(self):
        return "LearnableCategorical(n_classes={}, a_dim={})".format(self.n_classes, self.a_dim)
