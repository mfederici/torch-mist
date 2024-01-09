from torch_mist.critic import Critic
from torch_mist.baseline import (
    LearnableBaseline,
    InterpolatedBaseline,
    BatchLogMeanExp,
)
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)


class AlphaTUBA(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        baseline: LearnableBaseline,
        alpha: float = 0.01,
        neg_samples: int = -1,
    ):
        alpha_baseline = InterpolatedBaseline(
            baseline_1=BatchLogMeanExp("first"),
            baseline_2=baseline,
            alpha=alpha,
        )
        super().__init__(
            critic=critic,
            baseline=alpha_baseline,
            neg_samples=neg_samples,
        )
