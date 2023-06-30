from typing import List, Optional

import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.l1out import L1Out


class CLUB(L1Out):
    def __init__(
            self,
            q_Y_given_X: ConditionalDistribution,
            sample: str = 'all'  # Use all the off-diagonal samples as negative samples
    ):
        super().__init__(q_Y_given_X=q_Y_given_X)
        assert sample in ['all', 'one', 'l1o'], 'The sample must be one of \n' \
                                                '  "all": use all samples in the batch as negatives\n' \
                                                '  "one": use one sample in the batch as negative\n' \
                                                '  "l1o": use all off-diagonal samples in the batch as negatives'
        self.sample = sample

    def approx_log_p_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim == 2, 'The input must be 2D tensors'
        assert x.shape[0] == y.shape[0], 'The batch size of x and y must be the same'
        N = x.shape[0]

        q_Y_given_x = self.q_Y_given_x(x=x)

        if self.sample == 'one':
            y_ = torch.roll(y, 1, dims=0)
            log_p_y = q_Y_given_x.log_prob(y_)
            log_p_y = torch.roll(log_p_y, -1, 0)
        else:
            # Probability of all the other y in the same batch [N, N]
            log_p_y = q_Y_given_x.log_prob(y.unsqueeze(1))

            assert log_p_y.shape[1] == log_p_y.shape[0]

            if self.sample == 'l1o':
                # Remove the diagonal
                log_p_y = log_p_y * (1 - torch.eye(N).to(y.device))
                N = N - 1

            assert log_p_y.shape[1] == log_p_y.shape[0]

            log_p_y = torch.sum(log_p_y, dim=1)/N
        return log_p_y


def club(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        transform_name: str = 'conditional_linear',
        n_transforms: int = 1,
) -> CLUB:
    from torch_mist.distributions.utils import conditional_transformed_normal

    q_Y_given_X = conditional_transformed_normal(
        input_dim=y_dim,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
    )

    return CLUB(
        q_Y_given_X=q_Y_given_X,
    )
