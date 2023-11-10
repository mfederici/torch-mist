from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)
from torch_mist.critic import Critic
from torch_mist.baseline import ConstantBaseline


class NWJ(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
    ):
        super().__init__(
            critic=critic,
            neg_samples=neg_samples,
            baseline=ConstantBaseline(value=1.0),
        )
