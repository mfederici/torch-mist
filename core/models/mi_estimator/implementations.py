from typing import Optional, List

from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from core.models.marginals.transformed_normal import TransformedNormal
from core.models.learnable_distributions import ConditionalTransformedNormalProposal
from core.models.ratio import SeparableRatioEstimatorMLP
from core.models.ratio.base import RatioEstimator
from core.models.ratio.joint import JointRatioEstimatorMLP
from core.models.baseline.base import Baseline, ConstantBaseline, BatchLogMeanExp, ExponentialMovingAverage, \
    LearnableMLPBaseline, TUBABaseline, InterpolatedBaseline
from core.models.mi_estimator.base import MutualInformationEstimator, GenerativeMutualInformationEstimator


class BA(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            h_y: float,
            n_transforms: int,
            conditional_transform_name: str = "conditional_linear",
    ):
        proposal = ConditionalTransformedNormalProposal(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name=conditional_transform_name,
            n_transforms=n_transforms
        )

        super().__init__(
            proposal=proposal,
            h_y=h_y,
        )

class DoE(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            transform_name: str = "conditional_linear",
            n_transforms: int = 1,
            marginal_transform_name: str = "linear",
            n_marginal_transforms: int = 1,
    ):
        proposal = ConditionalTransformedNormalProposal(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name=transform_name,
            n_transforms=n_transforms
        )

        marginal_y = TransformedNormal(
            input_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name=marginal_transform_name,
            n_transforms=n_marginal_transforms
        )

        super().__init__(
            proposal=proposal,
            marginal_y=marginal_y,
            update_marginals=True
        )

class NWJ(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            neg_samples: int = 1,
            **kwargs

    ):
        ratio_estimator = JointRatioEstimatorMLP(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=ConstantBaseline(1.0),
            neg_samples=neg_samples,
            **kwargs
        )


class MINE(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            neg_samples: int = 1,
            ratio_estimator: Optional[RatioEstimator] = None,
            **kwargs
    ):
        if ratio_estimator is None:
            ratio_estimator = JointRatioEstimatorMLP(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dims=hidden_dims,
            )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=BatchLogMeanExp(dim=2),
            grad_baseline=ExponentialMovingAverage(gamma=0.9),
            neg_samples=neg_samples,
            **kwargs
        )


class InfoNCE(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            **kwargs
    ):
        ratio_estimator = SeparableRatioEstimatorMLP(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=BatchLogMeanExp(dim=1),
            neg_samples=0,  # 0 signifies the whole batch is used as negative samples
            **kwargs
        )


class JS(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            neg_samples: int = 1,
            **kwargs
    ):
        ratio_estimator = JointRatioEstimatorMLP(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=ConstantBaseline(1.0),
            js_grad=True,
            neg_samples=neg_samples,
            **kwargs
        )


class TUBA(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            baseline_hidden_dims: Optional[List[int]] = None,
            neg_samples: int = 1,
            **kwargs
    ):
        if baseline_hidden_dims is None:
            baseline_hidden_dims = hidden_dims

        ratio_estimator = JointRatioEstimatorMLP(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        )

        baseline = LearnableMLPBaseline(
            x_dim=x_dim,
            hidden_dims=baseline_hidden_dims,
        )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=baseline,
            neg_samples=neg_samples,
            **kwargs
        )


class AlphaTuba(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            baseline_hidden_dims: Optional[List[int]] = None,
            alpha: float = 0.5,
            neg_samples: int = 1,
            **kwargs,
    ):
        if baseline_hidden_dims is None:
            baseline_hidden_dims = hidden_dims

        ratio_estimator = JointRatioEstimatorMLP(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        )

        baseline_1 = BatchLogMeanExp()
        baseline_2 = LearnableMLPBaseline(x_dim, baseline_hidden_dims)

        baseline = InterpolatedBaseline(
            baseline_1=baseline_1,
            baseline_2=baseline_2,
            alpha=alpha
        )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=baseline,
            neg_samples=neg_samples,
            **kwargs,
        )


class SMILE(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            tau: float = 5.0,
            neg_samples: int = 1,
            **kwargs
    ):
        ratio_estimator = JointRatioEstimatorMLP(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        )

        super().__init__(
            ratio_estimator=ratio_estimator,
            baseline=BatchLogMeanExp(dim=2),
            js_grad=True,
            tau=tau,
            neg_samples=neg_samples,
            **kwargs
        )

