from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F

from core.models.mi_estimator.base import MutualInformationEstimator
from core.models.ratio import RatioEstimator
from core.models.baseline import Baseline, BatchLogMeanExp, ConstantBaseline, ExponentialMovingAverage, LearnableBaseline, \
    InterpolatedBaseline, LearnableJointBaseline


class DiscriminativeMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: Optional[RatioEstimator] = None,
            baseline: Optional[Baseline] = None,
            grad_baseline: Optional[Baseline] = None,
            neg_samples: int = 1,
            sample_gradient: bool = False,
            js_grad: bool = False,
            tau: Optional[float] = None,
    ):
        super().__init__()
        self.ratio_estimator = ratio_estimator
        self.baseline = baseline
        self.grad_baseline = grad_baseline
        self.neg_samples = neg_samples
        self.sample_gradient = sample_gradient
        self.js_grad = js_grad
        self.tau = tau

    def sample_marginals(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = y.shape[0]

        # For negative or zero values we consider N-self.n_samples instead
        if self.neg_samples <= 0:
            n_samples = N - self.neg_samples
        else:
            n_samples = self.neg_samples

        # If we use the whole batch as negatives
        if n_samples == N:
            # simply unsqueeze an empty dimension at the beginning
            return x, y[:, 0].unsqueeze(0)

        # Otherwise, take other ys in the batch as samples from the marginal
        # (excluding the diagonal if neg_samples < N)

        # This indexing operation takes care of selecting the appropriate (off-diagonal) y
        idx = torch.arange(N * n_samples).to(y.device).view(N, n_samples).long()
        idx = (idx % n_samples + torch.div(idx, n_samples, rounding_mode='floor') + 1) % N
        y_ = y[:, 0][idx]

        return x, y_

    @ staticmethod
    def _compute_ratio(x, y, f, f_, baseline):
        if baseline is not None:
            b = baseline(f_, x, y)
            if b.ndim == 1:
                b = b.unsqueeze(1)
            assert b.ndim == f_.ndim
        else:
            b = torch.zeros_like(x)

        f = f - b

        if isinstance(baseline, BatchLogMeanExp):
            Z = 1
        else:
            Z = f_.exp().mean(1).unsqueeze(1) / b.exp()

        ratio_value = f - Z + 1
        return ratio_value.mean()

    def _compute_ratio_value(self, x, y, f, f_):
        if not self.tau is None:
            f_ = torch.clamp(f_, min=-self.tau, max=self.tau)

        return self._compute_ratio(x, y, f, f_, self.baseline)

    def _compute_ratio_grad(self, x, y, f, f_):
        if self.js_grad:
            # Use the cross-entropy (with sigmoid predictive) to obtain the gradient
            ratio_grad = (- F.softplus(-f) - F.softplus(f_).mean(1).unsqueeze(1)).mean()
        elif self.grad_baseline is None:
            # The value and the gradient are the same
            ratio_grad = None
        else:
            # Use the second baseline to compute the gradient
            ratio_grad = self._compute_ratio(x, y, f, f_, self.grad_baseline)

        return ratio_grad

    def compute_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            x_: Optional[torch.Tensor] = None,
            y_: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        # Computation of gradient and value of E_{p(x,y)}[f(x,y)]-log E_{r(x,y)}[e^{f(x,y)}]
        estimates = {}

        # Compute the ratio f(x,y) on samples from p(x,y). The expected shape is [N, M]
        f = self.ratio_estimator(x, y)

        # Sample negatives from the product of the marginal distributions unless they are given
        if y_ is None:
            x_, y_ = self.sample_marginals(x, y)

        # Compute the ratio on the samples from the proposal [N, M']
        f_ = self.ratio_estimator(x, y_)

        ratio_value = self._compute_ratio_value(x, y, f, f_)
        if ratio_value is not None:
            estimates['value'] = ratio_value

        ratio_grad = self._compute_ratio_grad(x, y, f, f_)
        if ratio_grad is None:
            ratio_grad = ratio_value

        estimates['grad'] = ratio_grad

        return estimates


class NWJ(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            neg_samples: int = 1,
    ):

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=ConstantBaseline(1.0),
            neg_samples=neg_samples,
        )


class MINE(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            neg_samples: int = 1,
            gamma: float = 0.9,
    ):

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=BatchLogMeanExp(dim=2),
            grad_baseline=ExponentialMovingAverage(gamma=gamma),
            neg_samples=neg_samples,
        )


class InfoNCE(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            neg_samples: int = 0,
    ):
        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=BatchLogMeanExp(dim=1),
            neg_samples=neg_samples,  # 0 signifies the whole batch is used as negative samples
        )


class JS(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            neg_samples: int = 1,
    ):
        super().__init__(
            ratio_estimator=ratio_estimator,
            neg_samples=neg_samples,
            js_grad=True,
        )


class TUBA(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            baseline: LearnableBaseline,
            neg_samples: int = 1,
    ):
        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=baseline,
            neg_samples=neg_samples,
        )


class AlphaTUBA(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            baseline: Baseline = ConstantBaseline(1.0),
            alpha: float = 0.5,
            neg_samples: int = 1,
    ):

        baseline_1 = BatchLogMeanExp()

        alpha_baseline = InterpolatedBaseline(
            baseline_1=baseline_1,
            baseline_2=baseline,
            alpha=alpha
        )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=alpha_baseline,
            neg_samples=neg_samples,
        )


class SMILE(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            tau: float = 5.0,
            neg_samples: int = 1,
    ):

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=BatchLogMeanExp(dim=2),
            js_grad=True,
            tau=tau,
            neg_samples=neg_samples,
        )


class FLO(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            ratio_estimator: RatioEstimator,
            baseline: LearnableJointBaseline,
            neg_samples: int = 1,
    ):
        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=baseline,
            neg_samples=neg_samples,
        )

    def _compute_ratio(self, x, y, f, f_, baseline):
        b = baseline(f_, x, y)
        if b.ndim == 1:
            b = b.unsqueeze(1)
        assert b.ndim == f_.ndim

        Z = f_.exp().mean(1).unsqueeze(1) / (f - b).exp()

        ratio_value = b - Z + 1
        return ratio_value.mean()
