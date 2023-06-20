import torch
from torch import nn


class QuantizationFunction(nn.Module):
    @property
    def num_bins(self) -> int:
        raise NotImplemented()

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        raise NotImplemented()
