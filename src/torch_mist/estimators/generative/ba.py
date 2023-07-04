from typing import Optional, List, Union

import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import GenerativeMutualInformationEstimator
from torch_mist.utils.caching import reset_cache_after_call


class BA(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            q_Y_given_X: ConditionalDistribution,
            entropy_y: Optional[torch.Tensor] = None,
    ):
        super().__init__(q_Y_given_X=q_Y_given_X)
        if not isinstance(entropy_y, torch.Tensor):
            entropy_y = torch.tensor(entropy_y)
        entropy_y = entropy_y.squeeze()
        assert entropy_y.ndim == 0
        self.register_buffer('entropy_y', entropy_y)

    @reset_cache_after_call
    def expected_log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        entropy_y_given_x = -self.approx_log_p_y_given_x(x=x, y=y).mean()
        return self.entropy_y-entropy_y_given_x

def ba(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        entropy_y: Union[float, torch.Tensor],
        transform_name: str = 'conditional_linear',
        n_transforms: int = 1,
) -> BA:
    from torch_mist.distributions.utils import conditional_transformed_normal

    q_Y_given_X = conditional_transformed_normal(
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
        q_Y_given_X=q_Y_given_X,
        entropy_y=entropy_y,
    )





