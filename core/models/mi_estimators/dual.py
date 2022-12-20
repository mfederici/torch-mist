from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from core.models.baselines.base import ConstantBaseline, BatchLogMeanExp, ExponentialMovingAverage, LearnableJointBaseline

from core.models.mi_estimators.base import MutualInformationEstimator
from core.models.ratio.base import RatioEstimator


class NWJ(MutualInformationEstimator):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        MutualInformationEstimator.__init__(
            self,
            *args,
            baseline=ConstantBaseline(1),
            **kwargs
        )


class InfoNCE(MutualInformationEstimator):
    def __init__(
            self,
            *args,
            n_samples: int = 0,
            **kwargs,
    ):
        MutualInformationEstimator.__init__(
            self,
            *args,
            baseline=BatchLogMeanExp(),
            n_samples=n_samples,
            **kwargs
        )


class JS(NWJ):
    def compute_dual_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        # Use the cross-entropy (with sigmoid predictive) to obtain the gradient
        ratio_grad = (- F.softplus(-f) - F.softplus(f_).mean(1).unsqueeze(1))
        return ratio_grad.mean()


class MINE(MutualInformationEstimator):
    def __init__(
            self,
            *args,
            gamma: float = 0.9,
            **kwargs
    ):
        MutualInformationEstimator.__init__(
            self,
            *args,
            baseline=BatchLogMeanExp(dim=2),
            grad_baseline=ExponentialMovingAverage(gamma=gamma),
            **kwargs
        )


class SMILE(MINE, JS):
    def __init__(
            self,
            *args,
            tau: float = 5,
            **kwargs
    ):
        assert tau >= 0
        MINE.__init__(
            self,
            *args,
            **kwargs
        )
        self.tau = tau

    def compute_ratio_value(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        # Use a batch-based estimation of the Donsker-Varadhan bound:
        # log E_r(x,y)[e^f(x,y)] \approx logsumexp(f_) - log (M'*N)
        if self.tau is not None:
            f_ = torch.clamp(f_, -self.tau, self.tau)
        return MINE.compute_dual_ratio_value(self, x, y, f, f_)

    def compute_dual_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        return JS.compute_dual_ratio_grad(self, x, y, f, f_)


class FLO(MutualInformationEstimator):
    def __init__(
            self,
            *args,
            joint_baseline: nn.Module,
            **kwargs
    ):
        MutualInformationEstimator.__init__(
            self,
            *args,
            baseline=LearnableJointBaseline(joint_baseline),
            **kwargs
        )

    @staticmethod
    def _compute_dual_ratio_value(x, y, f, f_, baseline):
        b = baseline(f_, x, y)
        if b.ndim == 1:
            b = b.unsqueeze(1)
        assert b.ndim == f_.ndim

        Z = f_.exp().mean(1).unsqueeze(1) / (f-b).exp()

        ratio_value = b - Z + 1
        return ratio_value.mean()