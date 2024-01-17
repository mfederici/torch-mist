import math
from typing import Dict

import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import (
    ConditionalGenerativeMIEstimator,
)
from torch_mist.utils.indexing import select_k_others


class L1Out(ConditionalGenerativeMIEstimator):
    infomax_gradient: Dict[str, bool] = {"x": True, "y": False}

    def __init__(
        self, q_Y_given_X: ConditionalDistribution, neg_samples: int = -1
    ):
        super().__init__(q_Y_given_X=q_Y_given_X)
        self.neg_samples = neg_samples

    def _broadcast_log_p_y_given_x(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        assert (
            x.shape[0] == y.shape[0]
        ), "The batch size of x and y must be the same"
        N = x.shape[0]

        neg_samples = self.neg_samples
        if neg_samples <= 0:
            neg_samples = N + neg_samples

        if neg_samples < 1:
            neg_samples = 1

        if self.neg_samples == 0 or self.neg_samples == -1:
            x = x.unsqueeze(0)
        else:
            x = select_k_others(x, neg_samples)

        # Probability of all the other y in the same batch [M, N, ...]
        p_Y_given_x = self.q_Y_given_X.condition(x)
        log_p_y_given_x = p_Y_given_x.log_prob(y.unsqueeze(1) + x.detach() * 0)

        if self.neg_samples == -1:
            D = torch.eye(N).to(y.device)
            while D.ndim < log_p_y_given_x.ndim:
                D = D.unsqueeze(-1)

            # Remove the diagonal
            log_p_y_given_x = log_p_y_given_x * (1 - D)

            # Set the diagonal to -inf
            log_p_y_given_x = log_p_y_given_x + torch.nan_to_num(
                D * (-float("inf")),
                0,
                float("inf"),
                -float("inf"),
            )

        return log_p_y_given_x

    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # log prob of the samples in the batch, shape [M, N, ...]
        log_p_y_given_x = self._broadcast_log_p_y_given_x(x, y)

        # Compute the expectation using logsumexp. The shape is [N, ...]
        log_p_y = torch.logsumexp(log_p_y_given_x, dim=0) - math.log(
            log_p_y_given_x.shape[0]
        )

        assert log_p_y.shape == y.shape[:-1]

        return log_p_y
