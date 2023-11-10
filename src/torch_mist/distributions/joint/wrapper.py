from typing import Optional, List, Dict

import torch
from torch.distributions import Distribution

from torch_mist.distributions.joint.base import JointDistribution
from torch_mist.utils.shape import expand_to_same_shape


class TorchJointDistribution(JointDistribution):
    def __init__(
        self,
        torch_dist: Distribution,
        variables: List[str],
        splits: Optional[List[int]] = None,
        split_dim: int = -1,
        name: Optional[str] = "p",
    ):
        super().__init__(variables=variables, name=name)
        self.torch_dist = torch_dist
        if not (splits is None):
            assert len(splits) == len(variables)
        self.splits = splits
        self.split_dim = split_dim

    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.splits:
            tensors = torch.split(tensor, self.splits, dim=self.split_dim)
        else:
            tensors = torch.split(tensor, [1, 1], dim=self.split_dim)
            tensors = [tensor.squeeze(-1) for tensor in tensors]
        return {name: value for name, value in zip(self.variables, tensors)}

    def _dict_to_tensor(self, dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.splits is None:
            dict = {
                variable: tensor.unsqueeze(self.split_dim)
                for variable, tensor in dict.items()
            }
        return torch.cat(
            [dict[variable] for variable in self.variables], self.split_dim
        )

    def _log_prob(self, **kwargs) -> torch.Tensor:
        if len(self.variables) == 1:
            variable = self.variables[0]
            assert len(kwargs) == 1 and variable in kwargs
            tensor = kwargs[variable]
        else:
            assert len(kwargs) == len(
                self.variables
            ), f"passed: {kwargs.keys()} != required: {self.variables}"

            for label in kwargs:
                assert label in self.variables

            variables = []
            tensors = []
            for variable, tensor in kwargs.items():
                tensors.append(tensor)
                variables.append(variable)

            tensors = expand_to_same_shape(*tensors)
            tensor = self._dict_to_tensor(
                {variables[i]: tensors[i] for i in range(len(variables))}
            )

        # Compute the log prob
        log_prob = self.torch_dist.log_prob(tensor)
        return log_prob

    def _entropy(self, variables: List[str]) -> torch.Tensor:
        if len(variables) == 1 and len(self.variables) == 1:
            return self.torch_dist.entropy()
        else:
            raise NotImplementedError()

    def sample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Dict[str, torch.Tensor]:
        sample = self.torch_dist.sample(sample_shape)
        if len(self.variables) == 1:
            variable = self.variables[0]
            return {variable: sample}
        else:
            return self._tensor_to_dict(sample)

    def rsample(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Dict[str, torch.Tensor]:
        rsample = self.torch_dist.rsample(sample_shape)
        if len(self.variables) == 1:
            variable = self.variables[0]
            return {variable: rsample}
        else:
            return self._tensor_to_dict(rsample)
