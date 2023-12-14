from typing import Dict

import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.implementations.l1out import L1Out


class CLUB(L1Out):
    infomax_gradient: Dict[str, bool] = {"x": True, "y": False}

    def __init__(
        self, q_Y_given_X: ConditionalDistribution, neg_samples: int = 0
    ):
        super().__init__(q_Y_given_X=q_Y_given_X, neg_samples=neg_samples)

    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # log prob of the samples in the batch, shape [M, N, ...]
        log_p_y_given_x = self._broadcast_log_p_y_given_x(x, y)

        # Compute the expectation over M. The shape is [N, ...]
        log_p_y = torch.mean(log_p_y_given_x, dim=0)

        assert log_p_y.shape == y.shape[:-1]

        return log_p_y
