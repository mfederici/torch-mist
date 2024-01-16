from typing import Optional, Union
from functools import lru_cache

import torch
from torch.distributions import Distribution

from torch_mist.distributions.joint.base import JointDistribution
from torch_mist.estimators.generative.base import JointGenerativeMIEstimator
from torch_mist.utils.freeze import is_trainable


class GM(JointGenerativeMIEstimator):
    def __init__(
        self,
        q_XY: JointDistribution,
        q_Y: Union[Distribution, JointDistribution],
        q_X: Union[Distribution, JointDistribution],
    ):
        super().__init__(q_XY=q_XY)
        self._q_Y = q_Y
        self._q_X = q_X

    @property
    def q_X(self):
        return self._q_X

    @property
    def q_Y(self):
        return self._q_Y

    def batch_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        batch_loss = super().batch_loss(x=x, y=y)

        if is_trainable(self.q_Y):
            log_q_y = self.approx_log_p_y(x=x, y=y)
            batch_loss = batch_loss - log_q_y

        if is_trainable(self.q_X):
            log_q_x = self.approx_log_p_x(x=x, y=y)
            batch_loss = batch_loss - log_q_x

        assert (
            batch_loss.shape == y.shape[:-1] and torch.is_floating_point(y)
        ) or (batch_loss.shape == y.shape and not torch.is_floating_point(y))

        return batch_loss
