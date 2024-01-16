import math
from abc import abstractmethod

import torch
from torch import nn


class Baseline(nn.Module):
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()


class ConstantBaseline(Baseline):
    def __init__(self, value: float = 0):
        super(ConstantBaseline, self).__init__()
        self.register_buffer("value", torch.FloatTensor([value]))

    def forward(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        return self.value.expand(f_.shape[1:])


class ExponentialMovingAverage(Baseline):
    def __init__(self, gamma: float = 0.9):
        super(ExponentialMovingAverage, self).__init__()
        self.ema = None
        assert 0 <= gamma <= 1
        self.gamma = gamma

    def forward(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        average = f_.exp().mean().detach()
        average = torch.clamp(average, min=1e-4, max=1e4)

        if self.ema is None:
            ema = average
            self.ema = ema
        else:
            if self.training:
                ema = self.gamma * self.ema + (1 - self.gamma) * average
                self.ema = ema
            else:
                ema = self.ema

        ma = ema.log()
        for _ in range(x.ndim - 1):
            ma = ma.unsqueeze(0)
        ma = ma.expand_as(f_[0])
        return ma


class BatchLogMeanExp(Baseline):
    def __init__(self, dims: str):
        assert dims in ["first", "all"]
        super(BatchLogMeanExp, self).__init__()
        self.dims = dims

    def forward(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        # log 1/M \sum e^f_ = logsumexp(f_) - log M
        if self.dims == "all":
            f_ = f_.reshape(-1)

        M = f_.shape[0]
        b = torch.logsumexp(f_, 0) - math.log(M)

        if self.dims == "all":
            for _ in range(x.ndim - 1):
                b = b.unsqueeze(0)

        return b


class LearnableBaseline(Baseline):
    def __init__(self, net: nn.Module):
        super(LearnableBaseline, self).__init__()
        self.net = net

    def forward(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class InterpolatedBaseline(Baseline):
    def __init__(
        self, baseline_1: Baseline, baseline_2: Baseline, alpha: float
    ):
        super(InterpolatedBaseline, self).__init__()
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.baseline_1 = baseline_1
        self.baseline_2 = baseline_2

    def forward(
        self,
        x: torch.Tensor,
        f_: torch.Tensor,
    ) -> torch.Tensor:
        b1 = self.baseline_1.forward(f_=f_, x=x)
        b2 = self.baseline_2.forward(f_=f_, x=x)

        assert b1.shape == b2.shape

        if self.alpha == 0:
            return b2
        elif self.alpha == 1:
            return b1
        else:
            b = (self.alpha * b1.exp() + (1 - self.alpha) * b2.exp()).log()

            return b
