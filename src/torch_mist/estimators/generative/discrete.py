from typing import Optional

import torch
from torch import nn

from torch_mist.estimators.base import MutualInformationEstimator
from torch_mist.estimators.generative.gm import GM
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.caching import cached, reset_cache_after_call, reset_cache_before_call


class DiscreteMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            quantization_x: Optional[QuantizationFunction] = None,
            quantization_y: Optional[QuantizationFunction] = None,
            temperature: float = 0.1,
    ):
        self.x_bins = quantization_x.n_bins
        self.y_bins = quantization_y.n_bins
        super().__init__()

        self.quantization_x = quantization_x
        self.quantization_y = quantization_y

        # Start from uniform distribution
        self.logits_q_qxy = nn.Parameter(torch.zeros(self.x_bins, self.y_bins))

        self.temperature = temperature


    @cached
    def quantize_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantization_x is not None:
            x = self.quantization_x(x)
        else:
            assert isinstance(x, torch.LongTensor)
            x = torch.nn.functional.one_hot(x, self.x_bins).float()

        return x

    @cached
    def quantize_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.quantization_y is not None:
            y = self.quantization_y(y)
        else:
            assert isinstance(y, torch.LongTensor)
            y = torch.nn.functional.one_hot(y, self.y_bins).float()

        return y

    @property
    def log_q_qxy(self) -> torch.Tensor:
        log_q_qxy = self.logits_q_qxy / self.temperature - (self.logits_q_qxy.view(-1) / self.temperature).logsumexp(0)
        return log_q_qxy

    @cached
    def approx_log_p_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.quantize_x(x=x)
        y = self.quantize_y(y=y)

        # compute the empirical joint distribution
        xy = torch.einsum('...x, ...y -> ...xy', x, y)

        # Joint variational distribution
        log_q_xy = self.log_q_qxy

        # Add empty dimensions to log_q_xy
        while log_q_xy.ndim < xy.ndim:
            log_q_xy = log_q_xy.unsqueeze(0)

        log_q_xy = (log_q_xy*xy).sum(-1).sum(-1)
        assert log_q_xy.ndim == y.ndim-1
        return log_q_xy

    def approx_log_p_x(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.quantize_x(x=x)

        log_q_qx = torch.logsumexp(self.log_q_qxy, 1)

        while log_q_qx.ndim < x.ndim:
            log_q_qx = log_q_qx.unsqueeze(0)

        log_q_qx = (log_q_qx * x).sum(-1)
        assert log_q_qx.shape == x.shape[:-1], f'{log_q_qx.shape} != {x.shape[:-1]}'

        return log_q_qx

    def approx_log_p_y(self, y: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.quantize_y(y=y)

        log_q_qy = torch.logsumexp(self.log_q_qxy, 0)

        while log_q_qy.ndim < y.ndim:
            log_q_qy = log_q_qy.unsqueeze(0)

        log_q_qy = (log_q_qy * y).sum(-1)
        assert log_q_qy.shape == y.shape[:-1]
        return log_q_qy

    def log_p_x_given_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.approx_log_p_xy(x=x, y=y)-self.approx_log_p_y(y=y)

    def approx_log_p_y_given_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.approx_log_p_xy(x=x, y=y)-self.approx_log_p_x(x=x)

    @reset_cache_before_call
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = -self.approx_log_p_xy(x=x, y=y)
        return loss.mean()

    @reset_cache_after_call
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return GM.log_ratio(self, x=x, y=y)

def discrete(
        quantization_x: Optional[QuantizationFunction] = None,
        quantization_y: Optional[QuantizationFunction] = None,
) -> DiscreteMutualInformationEstimator:
    return DiscreteMutualInformationEstimator(
        quantization_x=quantization_x,
        quantization_y=quantization_y,
    )
