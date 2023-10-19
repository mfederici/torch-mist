from typing import Optional, List, Union

import torch

from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.base import (
    GenerativeMIEstimator,
)
from torch_mist.utils.caching import reset_cache_after_call


class BA(GenerativeMIEstimator):
    upper_bound: bool = True

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
        self.register_buffer("entropy_y", entropy_y)

    @reset_cache_after_call
    def expected_log_ratio(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        entropy_y_given_x = -self.approx_log_p_y_given_x(x=x, y=y).mean()
        return self.entropy_y - entropy_y_given_x


def ba(
    entropy_y: Union[float, torch.Tensor],
    x_dim: Optional[int] = None,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    q_Y_given_X: Optional[ConditionalDistribution] = None,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
) -> BA:
    from torch_mist.distributions.utils import conditional_transformed_normal

    if q_Y_given_X is None:
        if x_dim is None or y_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_Y_given_X or x_dim, y_dim and hidden_dims must be provided"
            )

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
