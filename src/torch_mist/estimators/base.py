from abc import abstractmethod
from typing import Dict, Union, Callable, Tuple, List

import torch
import torch.nn as nn


class MIEstimator(nn.Module):
    lower_bound: bool = False
    upper_bound: bool = False
    infomax_gradient: Dict[str, bool] = {"x": False, "y": False}

    def __init__(self):
        super().__init__()
        self._components_to_pretrain: List[Tuple[Callable, nn.Module]] = []

    @abstractmethod
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def mutual_information(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        log_ratio = self.log_ratio(x, y)
        assert (
            not isinstance(x, torch.LongTensor)
            and (log_ratio.shape == x.shape[:-1])
        ) or (isinstance(x, torch.LongTensor) and log_ratio.shape == x.shape)
        return log_ratio.mean()

    @abstractmethod
    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_loss = self.batch_loss(x=x, y=y)
        assert (
            (not isinstance(x, torch.LongTensor))
            and (batch_loss.shape == x.shape[:-1])
        ) or (isinstance(x, torch.LongTensor) and batch_loss.shape == x.shape)
        return batch_loss.mean()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.loss(x, y)
