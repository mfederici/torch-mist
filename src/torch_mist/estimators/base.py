from abc import abstractmethod

import torch
import torch.nn as nn


# The implementations in this work are loosely based on
# 1) "On Variational Lower bounds of mutual information" https://arxiv.org/pdf/1905.06922.pdf
# 2) "Undertanding the Limitations of Variational Mutual Information Estimators https://arxiv.org/abs/1910.06222


class MutualInformationEstimator(nn.Module):
    @abstractmethod
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def expected_log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_ratio(x, y).mean()

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

