from typing import Optional

import torch
from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import (
    ConditionalGenerativeMIEstimator,
)
from torch_mist.utils.caching import cached, reset_cache_before_call


class DoE(ConditionalGenerativeMIEstimator):
    def __init__(
        self,
        q_Y_given_X: ConditionalDistribution,
        q_Y: Distribution,
    ):
        super().__init__(
            q_Y_given_X=q_Y_given_X,
        )
        self.q_Y = q_Y

    @cached
    def approx_log_p_y(
        self, y: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        log_q_y = self.q_Y.log_prob(y)
        assert (
            log_q_y.shape == y.shape[:-1]
            and not isinstance(y, torch.LongTensor)
        ) or (log_q_y.shape == y.shape and isinstance(y, torch.LongTensor))
        return log_q_y

    def batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        log_q_y_given_x = self.approx_log_p_y_given_x(x=x, y=y)
        log_q_y = self.approx_log_p_y(x=x, y=y)

        loss = -log_q_y - log_q_y_given_x

        assert (
            loss.shape == y.shape[:-1] and not isinstance(y, torch.LongTensor)
        ) or (loss.shape == y.shape and isinstance(y, torch.LongTensor))

        return loss

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  "
            + "(q_Y_given_X): "
            + str(self.q_Y_given_X).replace("\n", "  \n")
            + "\n"
        )
        s += "  " + "(q_Y): " + str(self.q_Y).replace("\n", "  \n") + "\n"
        s += ")" + "\n"
        return s
