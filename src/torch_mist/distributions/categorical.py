import torch

from torch import nn
from torch.distributions import Distribution, Categorical

from torch_mist.distributions.parametrizations.map import LogitsMap
from torch_mist.distributions.transforms import ConditionalDistributionModule


class CategoricalModule(Distribution, nn.Module):
    def __init__(
        self,
        logits: torch.tensor,
        temperature: float = 1.0,
    ):
        nn.Module.__init__(self)
        Distribution.__init__(
            self,
            event_shape=torch.Size([logits.shape[0]]),
            validate_args=False,
        )

        self.logits = nn.Parameter(logits)
        self.parametrization = LogitsMap()
        self.temperature = temperature

    def rsample(self, sample_shape=torch.Size()):
        params = self.parametrization([self.logits / self.temperature])
        return Categorical(**params).rsample(sample_shape)

    def log_prob(self, value):
        params = self.parametrization([self.logits / self.temperature])
        return Categorical(**params).log_prob(value)

    def __repr__(self):
        return "Categorical()"


class ConditionalCategoricalModule(ConditionalDistributionModule):
    def __init__(self, net: nn.Module, temperature: float = 1.0):
        super(ConditionalCategoricalModule, self).__init__()
        self.net = net
        self.temperature = temperature
        self.parametrization = LogitsMap()

    def condition(self, x):
        return Categorical(
            **self.parametrization(self.net(x) / self.temperature)
        )
