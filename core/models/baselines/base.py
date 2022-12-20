import math
from abc import abstractmethod
from typing import Optional

import torch
from torch import nn


class Baseline(nn.Module):
    @abstractmethod
    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()


class ConstantBaseline(Baseline):
    def __init__(self, k: float = 0):
        super(ConstantBaseline, self).__init__()
        self.k = k

    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.zeros(f_.shape[0]).to(f_.device) + self.k


class ExponentialMovingAverage(Baseline):
    def __init__(self, gamma: float = 0.9):
        super(ExponentialMovingAverage, self).__init__()
        self.ema = None
        assert 0 <= gamma <= 1
        self.gamma = gamma

    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        average = f_.exp().mean().detach()
        average = torch.clamp(average, min=1e-4, max=1e4)

        if self.ema is None:
            ema = average
        else:
            ema = self.gamma * self.ema + (1 - self.gamma) * average

        if self.training:
            self.ema = ema

        ma = ema.log()

        return ma.unsqueeze(0).repeat(f_.shape[0])


class BatchLogMeanExp(Baseline):
    def __init__(self, dim=1):
        super(BatchLogMeanExp, self).__init__()
        self.dim = dim

    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, M = f_.shape[0], f_.shape[1]

        # log 1/M \sum_{j=1}^M f_[i,j] = logsumexp(f_, 1) - log M
        b = torch.logsumexp(f_, 1).unsqueeze(1) - math.log(M)

        if self.dim == 2:
            b = torch.logsumexp(b, 0).unsqueeze(0) - math.log(N)
        return b


class LearnableBaseline(Baseline):
    def __init__(self, net: nn.Module):
        super(LearnableBaseline, self).__init__()
        self.net = net

    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class LearnableJointBaseline(LearnableBaseline):
    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.unsqueeze(1)+y*0
        xy = torch.cat([x, y], -1)
        return self.net(xy).squeeze(-1)

class InterpolatedBaseline(LearnableBaseline):
    def __init__(self, net: nn.Module, alpha: float):
        super(InterpolatedBaseline, self).__init__(net=net)
        assert 0 <= alpha <= 1
        self.alpha = alpha

    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        M = f_.shape[0]
        # log 1/M \sum_{j=1}^M f_[i,j] = logsumexp(f_, 1) - log M
        m = torch.logsumexp(f_, 1) - math.log(M)
        a = super(InterpolatedBaseline, self).forward(f_, x)

        return self.alpha * m + (1 - self.alpha) * a
