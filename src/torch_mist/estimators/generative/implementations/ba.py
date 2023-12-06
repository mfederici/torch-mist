from typing import Optional

import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import (
    ConditionalGenerativeMIEstimator,
)
from torch_mist.utils.caching import (
    reset_cache_after_call,
    reset_cache_before_call,
)


class BA(ConditionalGenerativeMIEstimator):
    upper_bound: bool = True
    infomax_gradient: bool = True

    def __init__(
        self,
        q_Y_given_X: ConditionalDistribution,
        entropy_y: Optional[torch.Tensor] = None,
    ):
        super().__init__(q_Y_given_X=q_Y_given_X)
        if not isinstance(entropy_y, torch.Tensor):
            entropy_y = torch.tensor(entropy_y)
        entropy_y = entropy_y.squeeze()
        assert entropy_y.ndim == 0
        self.register_buffer("entropy_y", entropy_y)

    @reset_cache_after_call
    def mutual_information(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        entropy_y_given_x = -self.approx_log_p_y_given_x(x=x, y=y).mean()
        return self.entropy_y - entropy_y_given_x

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.approx_log_p_y_given_x(x=x, y=y)
