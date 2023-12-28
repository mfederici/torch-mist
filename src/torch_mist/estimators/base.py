from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class MIEstimator(nn.Module):
    lower_bound: bool = False
    upper_bound: bool = False
    infomax_gradient: Dict[str, bool] = {"x": False, "y": False}

    @abstractmethod
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.log_ratio(x, y)

    def mutual_information(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute an estimation for I(x,y).
        Args:
           x: a tensor with shape [N, D] in which x[i] is sampled from p(x)
           y: a tensor with shape [N, D] in which y[i] is sampled from p(y|x[i])
        Returns:
           estimation: the expected (weighted) log ratio
        """
        return self.mutual_information(x, y)
