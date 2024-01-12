from abc import abstractmethod
from functools import lru_cache

import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.distributions import JointDistribution
from torch_mist.distributions.cached import CachedConditionalDistribution
from torch_mist.estimators.base import MIEstimator


class GenerativeMIEstimator(MIEstimator):
    def unnormalized_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.log_ratio(x, y)

    @abstractmethod
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
        x_dim = x.ndim + (1 if isinstance(x, torch.LongTensor) else 0)
        y_dim = y.ndim + (1 if isinstance(y, torch.LongTensor) else 0)
        assert x_dim == y_dim, f"x.ndim={x_dim}, y.ndim={y_dim}"

        # Compute the ratio using the primal KL bound
        approx_log_p_y_given_x = self.approx_log_p_y_given_x(x=x, y=y)
        approx_log_p_y = self.approx_log_p_y(x=x, y=y)

        assert (
            approx_log_p_y_given_x.ndim == approx_log_p_y.ndim == x_dim - 1
        ), f"log_p_y_x.ndim={approx_log_p_y_given_x.ndim}, log_p_y.ndim={approx_log_p_y.ndim}"
        log_ratio = approx_log_p_y_given_x - approx_log_p_y

        return log_ratio


class ConditionalGenerativeMIEstimator(GenerativeMIEstimator):
    def __init__(self, q_Y_given_X: ConditionalDistribution):
        super().__init__()
        # Add caching to the conditioning for efficiency
        assert isinstance(q_Y_given_X, ConditionalDistribution)
        if not isinstance(q_Y_given_X, CachedConditionalDistribution):
            q_Y_given_X = CachedConditionalDistribution(q_Y_given_X)
        self.q_Y_given_X = q_Y_given_X

    @lru_cache(maxsize=1)
    def approx_log_p_y_given_x(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        q_Y_given_x = self.q_Y_given_X.condition(x)

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

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = -self.approx_log_p_y_given_x(x=x, y=y)

        assert (
            loss.shape == y.shape[:-1] and not isinstance(y, torch.LongTensor)
        ) or (
            loss.shape == y.shape and isinstance(y, torch.LongTensor)
        ), f"{isinstance(y, torch.FloatTensor)}. {loss.shape}!={y.shape[:-1]}"
        return loss

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


class JointGenerativeMIEstimator(GenerativeMIEstimator):
    def __init__(self, q_XY: JointDistribution):
        super().__init__()
        # Add caching to the conditioning for efficiency
        assert isinstance(q_XY, JointDistribution)
        self.q_XY = q_XY

    @property
    def q_X(self):
        return self.q_XY.marginal("x")

    @property
    def q_Y(self):
        return self.q_XY.marginal("y")

    @lru_cache(maxsize=1)
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

    @lru_cache(maxsize=1)
    def approx_log_p_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_q_x = self.q_X.log_prob(x)
        assert (
            log_q_x.shape == x.shape[:-1]
            and not isinstance(x, torch.LongTensor)
        ) or (log_q_x.shape == x.shape and isinstance(x, torch.LongTensor))
        # The shape is [...]
        return log_q_x

    @lru_cache(maxsize=1)
    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_q_y = self.q_Y.log_prob(y)
        assert (
            log_q_y.shape == y.shape[:-1]
            and not isinstance(y, torch.LongTensor)
        ) or (log_q_y.shape == y.shape and isinstance(y, torch.LongTensor))
        # The shape is [...]
        return log_q_y

    @lru_cache(maxsize=1)
    def approx_log_p_y_given_x(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        log_q_xy = self.approx_log_p_xy(x, y)
        log_q_x = self.approx_log_p_x(x, y)

        # Compute log q(Y=y|X=x)]
        log_q_y_given_x = log_q_xy - log_q_x

        assert (
            log_q_y_given_x.shape == y.shape[:-1]
            and not isinstance(y, torch.LongTensor)
        ) or (
            log_q_y_given_x.shape == y.shape
            and isinstance(y, torch.LongTensor)
        ), f"log_p_Y_X.shape={log_q_y_given_x.shape}, y.shape={y.shape}"

        return log_q_y_given_x

    def batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = -self.approx_log_p_xy(x=x, y=y)
        assert (
            loss.shape == y.shape[:-1] and isinstance(y, torch.FloatTensor)
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
