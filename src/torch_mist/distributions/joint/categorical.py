from typing import List, Dict, Optional

import numpy as np
import torch
from torch.distributions import Categorical
from torch import nn

from torch_mist.distributions.joint.base import JointDistribution, T


class JointCategorical(JointDistribution):
    def __init__(
        self,
        variables: List[str],
        bins: List[int],
        temperature: float = 1.0,
        logits: Optional[torch.Tensor] = None,
        name="p",
    ):
        super().__init__(variables=variables, name=name)
        if logits is None:
            self.logits = nn.Parameter(torch.zeros(np.prod(bins)))
        else:
            self.logits = logits

        self.bins = bins
        self.temperature = temperature

    @property
    def categorical(self) -> Categorical:
        return Categorical(logits=self.logits / self.temperature)

    def _tensor_to_dict(
        self, tensor: torch.LongTensor
    ) -> Dict[str, torch.LongTensor]:
        tensor_dict = {}
        for i in reversed(range(len(self.bins))):
            tensor_dict[self.variables[i]] = tensor % self.bins[i]
            tensor = tensor // self.bins[i]
        return tensor_dict

    def _dict_to_tensor(
        self, tensor_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        tensor = tensor_dict[self.variables[0]] + 0
        for i in range(1, len(self.variables)):
            tensor *= self.bins[i]
            tensor += tensor_dict[self.variables[i]]

        return tensor

    def _log_prob(self, **kwargs) -> torch.Tensor:
        if len(self.variables) == 1:
            variable = self.variables[0]
            assert len(kwargs) == 1 and variable in kwargs
            tensor = kwargs[variable]
        else:
            tensor = self._dict_to_tensor(kwargs)

        # Compute the log prob
        log_prob = self.categorical.log_prob(tensor)
        return log_prob

    def _entropy(self, variables: List[str]) -> torch.Tensor:
        return self.categorical.entropy()

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Dict[str, torch.Tensor]:
        sample = self.categorical.sample(sample_shape)
        return self._tensor_to_dict(sample)

    def _marginal(self: T, variables: List[str]) -> T:
        new_variables = []
        new_bins = []
        dims_to_remove = []
        for i, variable in enumerate(self.variables):
            if variable in variables:
                new_variables.append(variable)
                new_bins.append(self.bins[i])
            else:
                dims_to_remove.append(i)

        new_logits = torch.logsumexp(
            self.logits.view(*self.bins) / self.temperature, dim=dims_to_remove
        ).view(-1)

        return JointCategorical(
            variables=new_variables,
            bins=new_bins,
            logits=new_logits,
            temperature=1.0,
            name=self.name,
        )
