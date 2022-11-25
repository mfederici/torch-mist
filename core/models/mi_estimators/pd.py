from typing import Optional, Tuple, List

import torch
from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from core.distributions import conditional_transformed_normal
from core.models.ratio import RatioEstimator, JointRatioEstimator
from core.models.mi_estimators.dual import DualMutualInformationEstimator, JS, NWJ, MINE, SMILE, InfoNCE
from core.models.mi_estimators.primal import PrimalMutualInformationEstimator
from core.models.baselines import Baseline, ConstantBaseline, ExponentialMovingAverage, BatchLogMeanExp
from core.nn import JointDenseNN


class PrimalDualMutualInformationEstimator(PrimalMutualInformationEstimator, DualMutualInformationEstimator):
    def __init__(
        self,
        r_y_x: ConditionalDistribution,
        ratio_estimator: RatioEstimator,
        baseline: Optional[Baseline],
        n_samples: int = 1,
        p_y: Optional[Distribution] = None,
        h_y: Optional[float] = None,
        sample_gradient: bool = False
    ):

        PrimalMutualInformationEstimator.__init__(
            self,
            r_y_x=r_y_x,
            p_y=p_y,
            h_y=h_y
        )

        self.n_samples = n_samples
        self.ratio_estimator = ratio_estimator
        self.sample_gradient = sample_gradient
        self.baseline = baseline

    def compute_primal_ratio(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], torch.Tensor
    ]:
        return PrimalMutualInformationEstimator.compute_ratio(self, x, y)

    def compute_dual_ratio(self, x: torch.Tensor, y: torch.Tensor, y_: Optional[torch.Tensor]) -> Tuple[
        Optional[torch.Tensor], torch.Tensor
    ]:
        return DualMutualInformationEstimator.compute_ratio(self, x, y, y_)

    def sample_proposal(self, x, y):
        N = y.shape[0]
        # By default, we use the proposal is p(y)

        # For negative or zero values we consider N-self.n_samples instead
        if self.n_samples <= 0:
            n_samples = N - self.n_samples
        else:
            n_samples = self.n_samples

        r_y_X = self.r_y_x.condition(x)
        if self.sample_gradient:
            y_ = r_y_X.rsample([n_samples])
        else:
            y_ = r_y_X.sample([n_samples])

        return y_.transpose(0, 1)

    def compute_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        dual_value, dual_grad = self.compute_dual_ratio(x, y, y_)
        primal_value, primal_grad = self.compute_primal_ratio(x, y)

        assert primal_grad.shape == dual_grad.shape

        mi_grad = primal_grad + dual_grad
        if primal_value is not None:
            mi_value = primal_value + dual_value
        else:
            mi_value = None

        return mi_value, mi_grad

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '\n  (ratio_estimator): ' + str(self.ratio_estimator)
        s += '\n  (r_y_x): ' + self.r_y_x.__repr__()
        if self.p_y is not None:
            s += '\n  (p_y): ' + self.p_y.__repr__()
        s += '\n)'
        return s


class PNWJ(PrimalDualMutualInformationEstimator, NWJ):
    def __init__(self, *args, **kwargs):
        super(PNWJ, self).__init__(
            *args,
            baseline=ConstantBaseline(1),
            **kwargs
        )

    def compute_dual_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return NWJ.compute_ratio(self, x, y, y_)


class PJS(PNWJ):
    grad_is_value = False

    def compute_ratio_grad(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return JS.compute_ratio_grad(self, x, f, f_)


class PInfoNCE(PrimalDualMutualInformationEstimator, InfoNCE):
    def __init__(
            self,
            *args,
            n_samples=0,
            **kwargs,
    ):
        super(PInfoNCE, self).__init__(
            *args,
            baseline=BatchLogMeanExp(),
            n_samples=n_samples,
            **kwargs
        )


class PMINE(PrimalDualMutualInformationEstimator, MINE):
    grad_is_value = False

    def __init__(
            self,
            *args,
            gamma: float = 0.9,
            **kwargs
    ):
        super(PMINE, self).__init__(
            *args,
            baseline=ExponentialMovingAverage(gamma=gamma),
            **kwargs
        )

    def compute_dual_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return MINE.compute_ratio(self, x, y, y_)


class PSMILE(PrimalDualMutualInformationEstimator, SMILE):
    def __init__(
            self,
            *args,
            gamma: float = 0.9,
            tau : float = 5.,
            **kwargs
    ):
        super(PSMILE, self).__init__(
            *args,
            baseline=ExponentialMovingAverage(gamma=gamma),
            **kwargs
        )
        self.tau = tau

    def compute_ratio_grad(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        return SMILE.compute_ratio_grad(self, x, f, f_)

    def compute_ratio_value(
            self,
            x: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        return SMILE.compute_ratio_value(self, x, f, f_)


def pjs(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        transform_name: str = "conditional_linear",
        n_transforms: int = 1,
        p_y: Optional[Distribution] = None,
        h_y: Optional[float] = None,
        **transform_params
):
    dist = conditional_transformed_normal(
        input_dim=x_dim,
        context_dim=y_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
        **transform_params
    )
    ratio_estimator = JointRatioEstimator(
        joint_net=JointDenseNN(
            input_dims=[x_dim, y_dim],
            hidden_dims=hidden_dims,
            param_dims=[1]
        )
    )

    return PJS(ratio_estimator=ratio_estimator, r_y_x=dist, h_y=h_y, p_y=p_y)
