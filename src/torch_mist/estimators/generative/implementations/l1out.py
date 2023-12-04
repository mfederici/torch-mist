import math
import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import (
    ConditionalGenerativeMIEstimator,
)
from torch_mist.utils.caching import (
    reset_cache_after_call,
    reset_cache_before_call,
)
from torch_mist.utils.indexing import select_off_diagonal


class L1Out(ConditionalGenerativeMIEstimator):
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
            x = select_off_diagonal(x, neg_samples)

        # Probability of all the other y in the same batch [M, N, ...]
        p_y_given_X = self.q_Y_given_x(x=x)
        log_p_y_given_x = p_y_given_X.log_prob(y.unsqueeze(1))

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

    def log_ratio(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        log_p_y_x = self.approx_log_p_y_given_x(x=x, y=y)
        log_p_y = self.approx_log_p_y(x=x, y=y)

        assert (
            log_p_y_x.ndim == log_p_y.ndim
        ), f"log_p_y_x.ndim={log_p_y_x.ndim}, log_p_y.ndim={log_p_y.ndim}"
        log_ratio = log_p_y_x - log_p_y

        return log_ratio

    def batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        log_q_y_given_x = self.approx_log_p_y_given_x(x=x, y=y)
        loss = -log_q_y_given_x

        assert (
            loss.shape == y.shape[:-1] and isinstance(y, torch.FloatTensor)
        ) or (loss.shape == y.shape and isinstance(y, torch.LongTensor))

        return loss
