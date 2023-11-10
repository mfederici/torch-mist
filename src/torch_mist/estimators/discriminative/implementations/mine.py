from torch_mist.baseline import BatchLogMeanExp, ExponentialMovingAverage
from torch_mist.critic import Critic
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
    CombinedDiscriminativeMIEstimator,
)


class MINE(CombinedDiscriminativeMIEstimator):
    lower_bound = False  # Technically MINE is a lower bound, but sometimes it converges from above

    def __init__(
        self,
        critic: Critic,
        neg_samples: int = 1,
        gamma: float = 0.9,
    ):
        super().__init__(
            train_estimator=BaselineDiscriminativeMIEstimator(
                critic=critic,
                neg_samples=neg_samples,
                baseline=ExponentialMovingAverage(gamma=gamma),
            ),
            eval_estimator=BaselineDiscriminativeMIEstimator(
                critic=critic,
                neg_samples=neg_samples,
                baseline=BatchLogMeanExp("all"),
            ),
        )
