from typing import Optional

import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.base import MutualInformationEstimator, Estimation


class GenerativeMutualInformationEstimator(MutualInformationEstimator):
    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

    def log_prob_y_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        raise NotImplemented()

    def log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Estimation:
        # Compute the ratio using the primal bound
        estimates = {}
        if x.ndim < y.ndim:
            x = x.unsqueeze(1)
        assert x.ndim == y.ndim

        log_p_y_x = self.log_prob_y_x(x, y)  # [N, M]
        log_p_y = self.log_prob_y(x, y)  # [N, M]

        assert log_p_y_x.ndim == log_p_y.ndim == 2, f'log_p_y_x.ndim={log_p_y_x.ndim}, log_p_y.ndim={log_p_y.ndim}'

        value = log_p_y_x - log_p_y
        loss = self.compute_loss(x=x, y=y, log_p_y_x=log_p_y_x, log_p_y=log_p_y)

        return Estimation(value=value, loss=loss)


class VariationalProposalMutualInformationEstimator(GenerativeMutualInformationEstimator):
    def __init__(self, conditional_y_x: ConditionalDistribution):
        super().__init__()
        self.conditional_y_x = conditional_y_x

        self._cached_p_y_X = None
        self._cached_x = None
        self._cached_y = None

    def log_prob_y_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute E[log r(y|x)]
        p_y_X = self.conditional_y_x.condition(x)
        log_p_Y_X = p_y_X.log_prob(y)

        assert log_p_Y_X.shape == y.shape[:-1], f'log_p_Y_X.shape={log_p_Y_X.shape}, y.shape={y.shape}'

        # Cache the conditional p(y|X=x) and the inputs x, y
        self._cached_p_y_X = p_y_X
        self._cached_x = x
        self._cached_y = y

        return log_p_Y_X

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return -log_p_y_x.mean()

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(conditional_y_x): ' + str(self.conditional_y_x).replace('\n', '  \n') + '\n'
        s += ')' + '\n'

        return s
