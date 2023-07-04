from distutils.dist import Distribution
from typing import List
import math
import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import GenerativeMutualInformationEstimator
from torch_mist.utils.caching import cached, reset_cache_before_call, reset_cache_after_call


class L1Out(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            q_Y_given_X: ConditionalDistribution,
    ):
        super().__init__(q_Y_given_X=q_Y_given_X)

    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim == 2, 'The input must be 2D tensors'
        assert x.shape[0] == y.shape[0], 'The batch size of x and y must be the same'
        N = x.shape[0]

        p_y_given_X = self.q_Y_given_x(x=x)

        # Probability of all the other y in the same batch [N, N]
        log_p_y = p_y_given_X.log_prob(y.unsqueeze(1))

        assert log_p_y.shape[1] == log_p_y.shape[0]

        # Remove the diagonal
        log_p_y = log_p_y * (1 - torch.eye(N).to(y.device))

        # Set the diagonal to -inf
        log_p_y = log_p_y + torch.nan_to_num(
            torch.eye(N).to(y.device) * (-float('inf')),
            0, float('inf'), -float('inf')
        )

        # Compute the expectation using logsumexp. The shape is [N]
        log_p_y = torch.logsumexp(log_p_y, dim=1) - math.log(N - 1)
        return log_p_y

    def log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()

    @reset_cache_after_call
    def expected_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:

        log_p_y_x = self.approx_log_p_y_given_x(x=x, y=y)
        log_p_y = self.approx_log_p_y(x=x, y=y)

        assert log_p_y_x.ndim == log_p_y.ndim, f'log_p_y_x.ndim={log_p_y_x.ndim}, log_p_y.ndim={log_p_y.ndim}'
        log_ratio = log_p_y_x - log_p_y

        return log_ratio.mean()


def l1out(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        transform_name: str = 'conditional_linear',
        n_transforms: int = 1,
) -> L1Out:
    from torch_mist.distributions.utils import conditional_transformed_normal

    q_Y_given_X = conditional_transformed_normal(
        input_dim=y_dim,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
    )

    return L1Out(
        q_Y_given_X=q_Y_given_X,
    )
