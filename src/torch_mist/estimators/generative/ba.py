from typing import Optional, List, Union

import torch

from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import VariationalProposalMutualInformationEstimator


class BA(VariationalProposalMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            marginal_y: Optional[Distribution] = None,
            entropy_y: Optional[torch.Tensor] = None,
    ):
        super().__init__(conditional_y_x)
        self.conditional_y_x = conditional_y_x
        assert (marginal_y is None) ^ (
                entropy_y is None), 'Either the marginal distribution or the marginal entropy must be provided'

        self.marginal_y = marginal_y
        self.entropy_y = entropy_y

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.entropy_y is not None:
            return -torch.FloatTensor(self.entropy_y).unsqueeze(0).unsqueeze(1).to(y.device)
        else:
            log_q_y = self.marginal_y.log_prob(y)
            assert log_q_y.shape == y.shape[:-1]
            return log_q_y

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Optimize using maximum likelihood
        return -(log_p_y + log_p_y_x).mean()


def ba(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        entropy_y: Union[float, torch.Tensor],
        transform_name: str = 'conditional_linear',
        n_transforms: int = 1,
) -> BA:
    from torch_mist.distributions.utils import conditional_transformed_normal

    q_y_x = conditional_transformed_normal(
        input_dim=y_dim,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
    )

    if isinstance(entropy_y, float):
        entropy_y = torch.FloatTensor([entropy_y])

    entropy_y = entropy_y.squeeze()

    return BA(
        conditional_y_x=q_y_x,
        entropy_y=entropy_y,
    )





