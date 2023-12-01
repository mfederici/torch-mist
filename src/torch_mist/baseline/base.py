import math
from abc import abstractmethod
from typing import Optional, List

import torch
from torch import nn


class Baseline(nn.Module):
    @abstractmethod
    def forward(
        self,
        f_: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError()


class ConstantBaseline(Baseline):
    def __init__(self, value: float = 0):
        super(ConstantBaseline, self).__init__()
        self.value = value

    def forward(
        self,
        f_: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.zeros(f_.shape[1:]).to(f_.device) + self.value


class ExponentialMovingAverage(Baseline):
    def __init__(self, gamma: float = 0.9):
        super(ExponentialMovingAverage, self).__init__()
        self.ema = None
        assert 0 <= gamma <= 1
        self.gamma = gamma

    def forward(
        self,
        f_: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
        ma = ma.expand_as(x[..., 0])
        return ma


class BatchLogMeanExp(Baseline):
    def __init__(self, dims: str):
        assert dims in ["first", "all"]
        super(BatchLogMeanExp, self).__init__()
        self.dims = dims

    def forward(
        self,
        f_: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # log 1/M \sum e^f_ = logsumexp(f_) - log M
        if self.dims == "all":
            f_ = f_.reshape(-1)

        M = f_.shape[0]
        b = torch.logsumexp(f_, 0) - math.log(M)

        if self.dims == "all":
            for _ in range(x.ndim - 1):
                b = b.unsqueeze(0)
            b = b.expand_as(x[..., 0])

        return b


class LearnableBaseline(Baseline):
    def __init__(self, net: nn.Module):
        super(LearnableBaseline, self).__init__()
        self.net = net

    def forward(
        self,
        f_: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
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
        f_: torch.Tensor,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b1 = self.baseline_1.forward(f_=f_, x=x, y=y)
        b2 = self.baseline_2.forward(f_=f_, x=x, y=y)

        assert b1.shape == b2.shape

        if self.alpha == 0:
            return b2
        elif self.alpha == 1:
            return b1
        else:
            # We use logsumexp for numerical stability
            # log alpha * exp(b1) + (1-alpha) * exp(b2) = log(exp(b1+log alpha) + exp(b2+log(1-alpha)))
            b = torch.logsumexp(
                torch.cat(
                    [
                        b1.unsqueeze(-1) + math.log(self.alpha),
                        b2.unsqueeze(-1) + math.log(1 - self.alpha),
                    ],
                    -1,
                ),
                -1,
            )

            return b


class AlphaTUBABaseline(InterpolatedBaseline):
    def __init__(self, x_dim: int, hidden_dims: List[int], alpha: float):
        from torch_mist.baseline.factories import baseline_nn

        baseline_1 = BatchLogMeanExp("first")
        baseline_2 = baseline_nn(x_dim, hidden_dims)
        super(AlphaTUBABaseline, self).__init__(
            baseline_1=baseline_1, baseline_2=baseline_2, alpha=alpha
        )
