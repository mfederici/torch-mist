from typing import List

import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import VariationalProposalMutualInformationEstimator


class CLUB(VariationalProposalMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            sample: str = 'all'  # Use all the off-diagonal samples as negative samples
    ):
        super().__init__(conditional_y_x)
        assert sample in ['all', 'one', 'l1o'], 'The sample must be one of \n' \
                                                '  "all": use all samples in the batch as negatives\n' \
                                                '  "one": use one sample in the batch as negative\n' \
                                                '  "l1o": use all off-diagonal samples in the batch as negatives'
        self.sample = sample

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.equal(x, self._cached_x), 'The input x is not the same as the cached input x'
        assert torch.equal(y, self._cached_y), 'The input y is not the same as the cached input y'

        if self.sample == 'one':
            y_ = torch.roll(y, 1, dims=0)
            log_p_y = self._cached_p_y_X.log_prob(y_)
            log_p_y = log_p_y.mean(dim=0).unsqueeze(0)
        else:
            N = y.shape[0]
            assert y.shape[1] == 1, 'The CLUB estimator can only be used with a single y'
            y_ = y.squeeze(1).unsqueeze(0)
            log_p_y = self._cached_p_y_X.log_prob(y_)  # [N, N]

            if self.sample == 'l1o':
                # Remove the diagonal
                log_p_y = log_p_y * (1 - torch.eye(N).to(y.device))
                N = N - 1

            log_p_y = torch.sum(log_p_y, dim=0).unsqueeze(0)/N
        return log_p_y


def club(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        transform_name: str = 'conditional_linear',
        n_transforms: int = 1,
) -> CLUB:
    from torch_mist.distributions.utils import conditional_transformed_normal

    q_y_x = conditional_transformed_normal(
        input_dim=y_dim,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
    )

    return CLUB(
        conditional_y_x=q_y_x,
    )
