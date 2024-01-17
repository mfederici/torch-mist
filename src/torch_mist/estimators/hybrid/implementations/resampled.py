from typing import Optional, Tuple

import torch

from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.estimators.generative import ConditionalGenerativeMIEstimator
from torch_mist.estimators.generative.base import (
    JointGenerativeMIEstimator,
    GenerativeMIEstimator,
)
from torch_mist.estimators.hybrid.base import HybridMIEstimator
from torch_mist.utils.caching import cached_method


class ResampledHybridMIEstimator(HybridMIEstimator):
    def __init__(
        self,
        generative_estimator: GenerativeMIEstimator,
        discriminative_estimator: DiscriminativeMIEstimator,
    ):
        assert isinstance(
            generative_estimator, ConditionalGenerativeMIEstimator
        ) or isinstance(generative_estimator, JointGenerativeMIEstimator)
        super().__init__(generative_estimator, discriminative_estimator)

    @cached_method
    def sample_negatives(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        N = x.shape[0]
        neg_samples = self.n_negatives_to_use(N)

        # Sample from the conditional
        if isinstance(
            self.generative_estimator, ConditionalGenerativeMIEstimator
        ):
            # Replace the original proposal with the conditional q(y|x)
            self.proposal = self.generative_estimator.q_Y_given_X.condition(x)
            x_, y_, log_w = DiscriminativeMIEstimator.sample_negatives(
                self, x, y
            )

        # Sample from the joint distribution
        else:
            q_XY = self.generative_estimator.q_XY
            samples = q_XY.sample([neg_samples, *x.shape[:-1]])
            x_ = samples["x"]
            y_ = samples["y"]
            log_w = None

        return x_, y_, log_w
