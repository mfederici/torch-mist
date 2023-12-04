from abc import abstractmethod

import torch
import torch.nn as nn

from torch_mist.utils.caching import (
    reset_cache_before_call,
    reset_cache_after_call,
)


class MIEstimator(nn.Module):
    lower_bound: bool = False
    upper_bound: bool = False
    infomax_gradient: bool = False

    @abstractmethod
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @reset_cache_after_call
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

    @reset_cache_before_call
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
