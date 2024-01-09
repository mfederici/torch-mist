from typing import Optional, Dict

import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import (
    ConditionalGenerativeMIEstimator,
)


class BA(ConditionalGenerativeMIEstimator):
    lower_bound: bool = True
    infomax_gradient: Dict[str, bool] = {"x": True, "y": False}

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

    def mutual_information(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        entropy_y_given_x = -self.approx_log_p_y_given_x(x=x, y=y).mean()
        return self.entropy_y - entropy_y_given_x
