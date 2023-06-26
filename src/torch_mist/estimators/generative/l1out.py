from typing import List
import math
import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import VariationalProposalMutualInformationEstimator


class L1Out(VariationalProposalMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
    ):
        super().__init__(conditional_y_x)

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.equal(x, self._cached_x), 'The input x is not the same as the cached input x'
        assert torch.equal(y, self._cached_y), 'The input y is not the same as the cached input y'


        N = y.shape[0]
        assert y.shape[1] == 1, 'The L1Out estimator can only be used with a single y'
        y_ = y.squeeze(1).unsqueeze(0)
        log_p_y = self._cached_p_y_X.log_prob(y_)  # [N, N]

        # Remove the diagonal
        log_p_y = log_p_y * (1 - torch.eye(N).to(y.device))
        log_p_y = log_p_y + torch.nan_to_num(torch.eye(N).to(y.device)*(-float('inf')), 0, float('inf'), -float('inf'))
        log_p_y = torch.logsumexp(log_p_y, dim=0).unsqueeze(0) - math.log(N-1)
        return log_p_y



def l1out(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        transform_name: str = 'conditional_linear',
        n_transforms: int = 1,
) -> L1Out:
    from torch_mist.distributions.utils import conditional_transformed_normal

    q_y_x = conditional_transformed_normal(
        input_dim=y_dim,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
    )

    return L1Out(
        conditional_y_x=q_y_x,
    )
