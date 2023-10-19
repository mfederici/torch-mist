from abc import abstractmethod

import torch
import torch.nn as nn


class MIEstimator(nn.Module):
    lower_bound: bool = False
    upper_bound: bool = False

    @abstractmethod
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def expected_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.log_ratio(x, y).mean()

    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.log_ratio(x, y)

    @abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute an estimation for I(x,y).
        Args:
           x: a tensor with shape [N, D] in which x[i] is sampled from p(x)
           y: a tensor with shape [N, D] in which y[i] is sampled from p(y|x[i])
        Returns:
           estimation: the expected (weighted) log ratio
        """
        return self.expected_log_ratio(x, y)
