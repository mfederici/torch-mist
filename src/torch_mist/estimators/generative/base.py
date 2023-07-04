from abc import abstractmethod, ABC
from typing import Optional, List

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch_mist.estimators.base import MutualInformationEstimator
from torch_mist.utils.caching import cached, reset_cache_after_call, reset_cache_before_call


class GenerativeMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            q_Y_given_X: ConditionalDistribution
    ):
        super().__init__()
        self.q_Y_given_X = q_Y_given_X

    @cached
    def q_Y_given_x(self, x: torch.Tensor) -> Distribution:
        # q(Y|X=x)
        q_Y_given_x = self.q_Y_given_X.condition(x)

        return q_Y_given_x

    @cached
    def approx_log_p_y_given_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q_Y_given_x = self.q_Y_given_x(x=x)

        # Compute log q(Y=y|X=x)]
        log_q_y_given_x = q_Y_given_x.log_prob(y)

        assert log_q_y_given_x.ndim == y.ndim - 1, f'log_p_Y_X.shape={log_q_y_given_x.shape}, y.shape={y.shape}'
        return log_q_y_given_x

    @reset_cache_before_call
    def loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        log_q_y_x = self.approx_log_p_y_given_x(x=x, y=y)
        return -log_q_y_x.mean()

    @abstractmethod
    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        assert x.ndim == y.ndim, f'x.ndim={x.ndim}, y.ndim={y.ndim}'

        # Compute the ratio using the primal KL bound
        approx_log_p_y_given_x = self.approx_log_p_y_given_x(x=x, y=y)
        approx_log_p_y = self.approx_log_p_y(x=x, y=y)

        assert approx_log_p_y_given_x.ndim == approx_log_p_y.ndim == x.ndim - 1, \
            f'log_p_y_x.ndim={approx_log_p_y_given_x.ndim}, log_p_y.ndim={approx_log_p_y.ndim}'
        log_ratio = approx_log_p_y_given_x - approx_log_p_y

        return log_ratio

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(q_Y_given_X): ' + str(self.q_Y_given_X).replace('\n', '  \n') + '\n'
        s += ')' + '\n'

        return s
