import torch
from torch.distributions import Categorical, Independent
from torch import nn
from pyro.distributions import ConditionalDistribution


class ConditionalCategorical(ConditionalDistribution, nn.Module):
    def __init__(self, net: nn.Module):
        super(ConditionalCategorical, self).__init__()
        self.net = net

    def condition(self, x):
        logits = self.net(x)

        # Corner case to handle 1D
        if logits.ndim == x.ndim:
            logits = logits.unsqueeze(-2)

        return Categorical(logits=logits)
