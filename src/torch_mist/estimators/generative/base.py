from abc import abstractmethod

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.caching import (
    cached,
)


class GenerativeMIEstimator(MIEstimator):
    @cached
    def approx_log_p_y_given_x(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def log_ratio(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        xdim = x.ndim + (1 if isinstance(x, torch.LongTensor) else 0)
        ydim = y.ndim + (1 if isinstance(y, torch.LongTensor) else 0)
        assert xdim == ydim, f"x.ndim={xdim}, y.ndim={ydim}"

        # Compute the ratio using the primal KL bound
        approx_log_p_y_given_x = self.approx_log_p_y_given_x(x=x, y=y)
        approx_log_p_y = self.approx_log_p_y(x=x, y=y)

        assert (
            approx_log_p_y_given_x.ndim == approx_log_p_y.ndim == x.ndim - 1
        ), f"log_p_y_x.ndim={approx_log_p_y_given_x.ndim}, log_p_y.ndim={approx_log_p_y.ndim}"
        log_ratio = approx_log_p_y_given_x - approx_log_p_y

        return log_ratio


class ConditionalGenerativeMIEstimator(GenerativeMIEstimator):
    def __init__(self, q_Y_given_X: ConditionalDistribution):
        super().__init__()
        self.q_Y_given_X = q_Y_given_X

    @cached
    def q_Y_given_x(self, x: torch.Tensor) -> Distribution:
        # q(Y|X=x)
        q_Y_given_x = self.q_Y_given_X.condition(x)

        return q_Y_given_x

    @cached
    def approx_log_p_y_given_x(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        q_Y_given_x = self.q_Y_given_x(x=x)

        # Compute log q(Y=y|X=x)]
        log_q_y_given_x = q_Y_given_x.log_prob(y)

        assert (
            log_q_y_given_x.shape == y.shape[:-1]
            and not isinstance(y, torch.LongTensor)
        ) or (
            log_q_y_given_x.shape == y.shape
            and isinstance(y, torch.LongTensor)
        ), f"log_p_Y_X.shape={log_q_y_given_x.shape}, y.shape={y.shape}"

        return log_q_y_given_x

    @abstractmethod
    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        s += (
            "  "
            + "(q_Y_given_X): "
            + str(self.q_Y_given_X).replace("\n", "  \n")
            + "\n"
        )
        s += ")" + "\n"

        return s
