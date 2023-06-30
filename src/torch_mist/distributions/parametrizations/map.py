import inspect
from abc import abstractmethod
from typing import Optional, Type, Dict, List, Union

import torch
from pyro.distributions import Delta
from torch.distributions import (
    Distribution,
    Normal,
    Laplace,
    Categorical,
    OneHotCategorical,
)
from torch.nn.functional import softplus

EPSILON = 1e-6


class ParameterMap:
    """
    Mapping from a list of tensors to a dictionary representing the parametrization of a distribution.
    Abstract class.
    """

    supported_distributions: List[Type[Distribution]] = []
    n_parameters: int = 0

    @abstractmethod
    def map_parameters(self, parameter_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Abstract method responsible for defining the mapping from the list of tensors to the parameter dictionary.

        Args:
            parameter_list (List[torch.Tensor]): The list of parameters for the distribution of interest.

        Returns:
            Dict[str, torch.Tensor]: The named parameters for the distribution of interest.
        """
        pass

    def __call__(
        self, parameter_list: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(parameter_list, torch.Tensor):
            parameter_list = [parameter_list]

        assert len(parameter_list) == self.n_parameters

        mapped_parameters = self.map_parameters(parameter_list)

        # Check the mapping is compatible with the specified distributions
        for distribution_class in self.supported_distributions:
            for parameter_name in mapped_parameters:
                assert parameter_name in inspect.signature(distribution_class).parameters

        return mapped_parameters


class DeltaMap(ParameterMap):
    """
    Delta Parameter mapping: maps a list of one tensor into a delta distribution at the same location
    """

    supported_distributions: List[Type[Distribution]] = [Delta]

    n_parameters: int = 1

    def map_parameters(self, parameter_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"v": parameter_list[0]}


class LocScaleMap(ParameterMap):
    supported_distributions: List[Type[Distribution]] = [Normal, Laplace]

    def __init__(self, scale: Optional[float] = None, epsilon: float = EPSILON, scale_offset: float = 0.0):
        """Location and scale parametrization that can be used for Normal and Laplace distributions. The scale is constrained to be positive. This is obtained by applying a softplus function.

        Args:
            scale (Optional[float], optional): The fixed scale of the distribution. If passed, only the tensor list maps to the scale only. Defaults to None.
            epsilon (float, optional): Small float used to increased the numerical stability. Defaults to EPSILON.
        """
        if scale:
            n_parameters = 1
        else:
            n_parameters = 2

        self.scale = scale
        self.epsilon = epsilon
        self.n_parameters = n_parameters
        self.scale_offset = scale_offset

    def map_parameters(self, parameter_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        loc = parameter_list[0]
        if self.n_parameters == 1:
            scale = self.scale
        else:
            log_scale = parameter_list[1]
            # Numerically stable scale parametrization
            scale = softplus(log_scale+self.scale_offset) + self.epsilon
            assert torch.all(scale > 0), f"Scale must be positive, got {scale}"

        return {"loc": loc, "scale": scale}


class LogitsMap(ParameterMap):
    supported_distributions: List[Type[Distribution]] = [
        Categorical,
        OneHotCategorical,
    ]
    n_parameters: int = 1

    def map_parameters(self, parameter_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        logits = parameter_list[0]
        return {"logits": logits}
