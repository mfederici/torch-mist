from torch_mist.baseline import LearnableBaseline
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)


class TUBA(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        baseline: LearnableBaseline,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            baseline=baseline,
            neg_samples=neg_samples,
        )
