import torch

from torch_mist.baseline import BatchLogMeanExp
from torch_mist.critic import SeparableCritic
from torch_mist.estimators.discriminative.base import (
    BaselineDiscriminativeMIEstimator,
)


class InfoNCE(BaselineDiscriminativeMIEstimator):
    def __init__(
        self,
        critic: SeparableCritic,
    ):
        # Note that this can be equivalently obtained by extending TUBA with a BatchLogMeanExp(dim=1) baseline
        # This implementation saves some computation
        super().__init__(
            critic=critic,
            neg_samples=0,  # 0 signifies the whole batch is used as negative samples
            baseline=BatchLogMeanExp("first"),
        )

    def batch_approx_log_partition(
        self, x: torch.Tensor, y: torch.Tensor, f_: torch.tensor
    ):
        # We override the compute_log_normalization just for efficiency
        # The result would be the same as the TUBA implementation with BatchLogMeanExp('first') baseline
        # We override the compute_log_normalization for efficiency since e^(F(x,y))-b(x) = 1
        log_norm = self.baseline(x=x, y=y, f_=f_).unsqueeze(0).expand(f_.shape)
        return log_norm
