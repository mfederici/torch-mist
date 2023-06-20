from abc import abstractmethod
from typing import Tuple, Optional, Dict, Union, Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn


# The implementations in this work are loosely based on
# 1) "On Variational Lower bounds of mutual information" https://arxiv.org/pdf/1905.06922.pdf
# 2) "Undertanding the Limitations of Variational Mutual Information Estimators https://arxiv.org/abs/1910.06222

@dataclass
class Estimation:
    value: torch.Tensor
    loss: torch.Tensor

    def __add__(self, other):
        return Estimation(self.value + other.value, self.loss + other.loss)

    def __sub__(self, other):
        return Estimation(self.value - other.value, self.loss - other.loss)

    def apply(self, fn):
        return Estimation(fn(self.value), fn(self.loss))

    @property
    def shape(self):
        if isinstance(self.value, float):
            return torch.Size([])
        else:
            return self.value.shape


class LogRatioEstimator(nn.Module):

    @abstractmethod
    def log_ratio(self, *args, **kwargs) -> Estimation:
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> Estimation:
        return self.log_ratio(*args, **kwargs)


class MutualInformationEstimator(LogRatioEstimator):
    lower_bound: bool
    upper_bound: bool

    def forward(self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Estimation:
        """
        Compute an estimation for I(x,y).
        Args:
            x: a tensor with shape [N, D] in which x[i] is sampled from p(x)
            y: a tensor with shape [N, D] or [N, M, D] in which y[i,j] is sampled from p(y|x[i])
        Returns:
            estimation: an object with the following attributes:
                        'value': the estimation for I(x,y)
                        'loss': a quantity to differentiate to maximize mutual information
                            w.r.t proposal, encoder(s) and ratio estimator
                        ...
                        other quantities that are computed by components
        """

        if y.ndim == x.ndim:
            # If one dimension is missing, we assume there is only one positive sample
            y = y.unsqueeze(1)

        assert y.ndim == x.ndim + 1

        # Compute the expected log ratio
        estimates = self.log_ratio(x=x, y=y)

        # Return the average ratio and loss
        return estimates.apply(lambda v: v.mean())

