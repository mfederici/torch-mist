from typing import Optional, Union

import torch
from torch.distributions import Distribution

from torch_mist.distributions.joint.base import JointDistribution
from torch_mist.estimators import MIEstimator
from torch_mist.utils.caching import (
    cached,
    reset_cache_after_call,
    reset_cache_before_call,
)


class GM(MIEstimator):
    def __init__(
        self,
        q_XY: JointDistribution,
        q_Y: Optional[Union[Distribution, JointDistribution]] = None,
        q_X: Optional[Union[Distribution, JointDistribution]] = None,
    ):
        super().__init__()
        self.q_XY = q_XY
        self._q_Y = q_Y
        self._q_X = q_X

    @property
    def q_X(self):
        if self._q_X:
            return self._q_X
        else:
            return self.q_XY.marginal("x")

    @property
    def q_Y(self):
        if self._q_Y:
            return self._q_Y
        else:
            return self.q_XY.marginal("y")

    @cached
    def approx_log_p_x(self, x: torch.Tensor) -> torch.Tensor:
        log_q_x = self.q_X.log_prob(x)
        assert (
            log_q_x.shape == x.shape[:-1]
            and not isinstance(x, torch.LongTensor)
        ) or (log_q_x.shape == x.shape and isinstance(x, torch.LongTensor))
        # The shape is [...]
        return log_q_x

    @cached
    def approx_log_p_y(self, y: torch.Tensor) -> torch.Tensor:
        log_q_y = self.q_Y.log_prob(y)
        assert (
            log_q_y.shape == y.shape[:-1]
            and not isinstance(y, torch.LongTensor)
        ) or (log_q_y.shape == y.shape and isinstance(y, torch.LongTensor))
        # The shape is [...]
        return log_q_y

    @cached
    def approx_log_p_xy(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # The shape is [...]
        log_q_xy = self.q_XY.log_prob(x=x, y=y)

        assert (
            log_q_xy.shape == y.shape[:-1]
            and not isinstance(y, torch.LongTensor)
        ) or (log_q_xy.shape == y.shape and isinstance(y, torch.LongTensor))

        return log_q_xy

    @reset_cache_after_call
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_q_xy = self.approx_log_p_xy(x=x, y=y)
        log_q_y = self.approx_log_p_y(y=y)
        log_q_x = self.approx_log_p_x(x=x)

        mi = log_q_xy - log_q_y - log_q_x

        assert (
            mi.shape == y.shape[:-1] and not isinstance(y, torch.LongTensor)
        ) or (mi.shape == y.shape and isinstance(y, torch.LongTensor))

        return mi

    @reset_cache_before_call
    def batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        log_q_xy = self.approx_log_p_xy(x=x, y=y)
        log_q_y = self.approx_log_p_y(y=y)
        log_q_x = self.approx_log_p_x(x=x)

        loss = -log_q_xy - log_q_y - log_q_x
        assert (
            loss.shape == y.shape[:-1] and not isinstance(y, torch.LongTensor)
        ) or (loss.shape == y.shape and isinstance(y, torch.LongTensor))

        return loss

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += "  (q_XY): " + str(self.q_XY).replace("\n", "\n  ") + "\n"
        if self.q_X is not None:
            s += "  (q_X): " + str(self.q_X).replace("\n", "\n  ") + "\n"
        if self.q_Y is not None:
            s += "  (q_Y): " + str(self.q_Y).replace("\n", "\n  ") + "\n"
        s += ")" + "\n"
        return s
