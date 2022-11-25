from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.models.baselines.base import Baseline, ConstantBaseline, BatchLogMeanExp, ExponentialMovingAverage

from core.models.mi_estimators.base import MutualInformationEstimator
from core.models.ratio.base import RatioEstimator


# from core.models.ratio.joint import JointRatioEstimator
# from core.models.ratio.separable import SeparableRatioEstimator
# from core.nn import JointDenseNN
# from core.logging import LoggingModule


# Mutual Information Estimation Based on the dual representation of a KL divergence
class DualMutualInformationEstimator(MutualInformationEstimator):
    grad_is_value: bool = True

    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            baseline: Optional[Baseline],
            n_samples: Optional[int] = 1,
    ):
        MutualInformationEstimator.__init__(self)

        self.ratio_estimator = ratio_estimator
        self.n_samples = n_samples
        self.baseline = baseline

    def sample_proposal(self, x, y) -> torch.Tensor:
        N = y.shape[0]
        # By default, we use the proposal is p(y)

        # For negative or zero values we consider N-self.n_samples instead
        if self.n_samples <= 0:
            n_samples = N-self.n_samples
        else:
            n_samples = self.n_samples

        if n_samples == N:
            # TODO: we can remove this assertion and pick the first y instead
            assert y.shape[1] == 1
            # Consider the whole batch as negatives
            y_ = y[:,0].unsqueeze(0).repeat(N, 1, 1)
        elif self.n_samples == 1:
            # Since the samples are iid, by shifting the pairing by 1 we can obtain samples from p(y) instead of p(y|x)
            y_ = torch.roll(y, 1, 0)
        else:
            assert self.n_samples <= N
            raise NotImplementedError()
            # TODO: write this as a convolution with an identity kernel with size [1,N,N,D] to obtain [1,N,D]
        return y_

    @abstractmethod
    def compute_ratio_value(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def compute_ratio_grad(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def compute_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # Computation of gradient and value of E_{p(x,y)}[f(x,y)]-log E_{r(x,y)}[e^{f(x,y)}]

        # Compute the ratio f(x,y) on samples from p(x,y). The expected shape is [N, M]
        f = self.ratio_estimator(x, y)

        if y_ is None:
            # Produce the specified number of samples y_ (with shape [N,M',D]) from the proposal r(y|x)
            y_ = self.sample_proposal(x, y)

        # Compute the ratio on the samples from the proposal [N, M']
        f_ = self.ratio_estimator(x, y_)

        ratio_value = self.compute_ratio_value(x, f, f_)

        if self.grad_is_value:
            ratio_grad = ratio_value
        else:
            ratio_grad = self.compute_ratio_grad(x, f, f_)

        return ratio_value, ratio_grad

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '\n  (ratio_estimator): ' + str(self.ratio_estimator)
        s += '\n)'
        return s


# TUBA estimator https://arxiv.org/pdf/1905.06922.pdf
class TUBA(DualMutualInformationEstimator):
    def compute_ratio_value(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor,
    ) -> torch.Tensor:
        # Use the bound log E_r[e^f(x,y)] <= E_{r(x,y)}[e^{f(x,y)-a(x)}] + E_{r(x)}[a(x)]] - 1

        # Compute the value of the baseline (shape [N])
        a = self.baseline(f_, x)
        assert a.shape[0] == x.shape[0]
        assert a.ndim == 1
        a = a.unsqueeze(1)
        assert a.ndim == f_.ndim

        if isinstance(self.baseline, BatchLogMeanExp):
            # For InfoNCE (f_-a).exp().mean() is = 1, so we can skip the computation
            log_Z = 1
        else:
            log_Z = (f_ - a).exp().mean()

        log_Z += a.mean() - 1

        ratio_value = f - log_Z

        return ratio_value

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '\n  (ratio_estimator): ' + str(self.ratio_estimator)
        s += '\n  (baseline): ' + str(self.baseline)
        s += '\n)'
        return s


class NWJ(TUBA):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            n_samples: int = 1,
    ):
        TUBA.__init__(
            self,
            ratio_estimator=ratio_estimator,
            baseline=ConstantBaseline(1),
            n_samples=n_samples,
        )


class InfoNCE(TUBA):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
    ):
        TUBA.__init__(
            self,
            ratio_estimator=ratio_estimator,
            baseline=BatchLogMeanExp(),
            n_samples=0,
        )


class JS(NWJ):
    grad_is_value: bool = False

    def compute_ratio_grad(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use the cross-entropy (with sigmoid predictive) to obtain the gradient
        mi_grad = (- F.softplus(-f) - F.softplus(f_).mean(1).unsqueeze(1))
        return mi_grad


class MINE(TUBA):
    grad_is_value = False

    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            gamma: float = 0.9,
            n_samples: int = 1,

    ):
        super(MINE, self).__init__(
            ratio_estimator=ratio_estimator,
            baseline=ExponentialMovingAverage(gamma=gamma),
            n_samples=n_samples,
        )

    def compute_ratio_value(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        # Use a batch-based estimation of the Donsker-Varadhan bound:
        # log E_r(x,y)[e^f(x,y)] \approx logsumexp(f_) - log (M'*N)
        mi_value = f - torch.logsumexp(f_, (0, 1)) + np.log(f_.numel())
        return mi_value

    def compute_ratio_grad(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        return super(MINE, self).compute_ratio_value(x, f, f_)


class SMILE(MINE):
    grad_is_value = False

    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            n_samples: int = 1,
            tau: float = None

    ):
        MINE.__init__(
            self,
            ratio_estimator=ratio_estimator,
            n_samples=n_samples,
        )
        self.tau = tau

    def compute_ratio_value(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        # Use a batch-based estimation of the Donsker-Varadhan bound:
        # log E_r(x,y)[e^f(x,y)] \approx logsumexp(f_) - log (M'*N)
        if self.tau is not None:
            f_ = torch.clamp(f_, -self.tau, self.tau)
        return MINE.compute_ratio_value(self, x, f, f_)

    def compute_ratio_grad(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        # Use the cross-entropy (with sigmoid predictive) to obtain the gradient
        mi_grad = - F.softplus(-f) - F.softplus(f_).mean(1).unsqueeze(1)
        return mi_grad