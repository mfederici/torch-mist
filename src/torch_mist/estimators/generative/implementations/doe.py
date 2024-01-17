from typing import Optional, Dict

import torch
from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import (
    ConditionalGenerativeMIEstimator,
)
from torch_mist.utils.caching import cached_method
from torch_mist.utils.freeze import is_trainable


class DoE(ConditionalGenerativeMIEstimator):
    infomax_gradient: Dict[str, bool] = {"x": True, "y": False}

    def __init__(
        self,
        q_Y_given_X: ConditionalDistribution,
        q_Y: Distribution,
    ):
        super().__init__(
            q_Y_given_X=q_Y_given_X,
        )
        self.q_Y = q_Y

    @cached_method
    def approx_log_p_y(
        self, y: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        log_q_y = self.q_Y.log_prob(y)
        assert (
            log_q_y.shape == y.shape[:-1] and torch.is_floating_point(y)
        ) or (log_q_y.shape == y.shape and not torch.is_floating_point(y))
        return log_q_y

    def batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        batch_loss = super().batch_loss(x, y)
        if is_trainable(self.q_Y):
            log_q_y = self.approx_log_p_y(x=x, y=y)
            batch_loss = batch_loss - log_q_y

            assert (
                batch_loss.shape == y.shape[:-1] and torch.is_floating_point(y)
            ) or (
                batch_loss.shape == y.shape and not torch.is_floating_point(y)
            )

        return batch_loss

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
