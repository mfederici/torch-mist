import math
from abc import abstractmethod
from typing import Optional, List

import torch
from pyro.nn import DenseNN
from torch import nn


class Baseline(nn.Module):
    @abstractmethod
    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()


class ConstantBaseline(Baseline):
    def __init__(self, value: float = 0):
        super(ConstantBaseline, self).__init__()
        self.value = value

    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.zeros(f_.shape[0]).to(f_.device) + self.value


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


class LearnableMLPBaseline(LearnableBaseline):
    def __init__(self, x_dim: int, hidden_dims: List[int]):
        net = DenseNN(x_dim, hidden_dims, [1])
        super(LearnableMLPBaseline, self).__init__(net)


class LearnableJointBaseline(LearnableBaseline):
    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.unsqueeze(1)+y*0
        xy = torch.cat([x, y], -1)
        return self.net(xy).squeeze(-1)


class InterpolatedBaseline(Baseline):
    def __init__(self, baseline_1: Baseline, baseline_2: Baseline, alpha: float):
        super(InterpolatedBaseline, self).__init__()
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.baseline_1 = baseline_1
        self.baseline_2 = baseline_2

    def forward(self, f_: torch.Tensor, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        b1 = self.baseline_1(f_, x, y)
        b2 = self.baseline_2(f_, x, y)
        return self.alpha * b1 + (1 - self.alpha) * b2


class TUBABaseline(InterpolatedBaseline):
    def __init__(self, x_dim: int, hidden_dims: List[int], alpha: float):
        baseline_1 = LearnableMLPBaseline(x_dim, hidden_dims)
        baseline_2 = BatchLogMeanExp()
        super(TUBABaseline, self).__init__(baseline_1=baseline_1, baseline_2=baseline_2, alpha=alpha)

