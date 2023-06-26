from typing import List

from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from torch_mist.estimators.generative.ba import BA


class DoE(BA):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            marginal_y: Distribution,
    ):
        super().__init__(
            conditional_y_x=conditional_y_x,
            marginal_y=marginal_y,
        )

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(conditional_y_x): ' + str(self.conditional_y_x).replace('\n', '  \n') + '\n'
        s += '  ' + '(marginal_y): ' + str(self.marginal_y).replace('\n', '  \n') + '\n'
        s += ')' + '\n'

        return s

def doe(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        conditional_transform_name: str = 'conditional_linear',
        n_conditional_transforms: int = 1,
        marginal_transform_name: str = 'linear',
        n_marginal_transforms: int = 1,
) -> DoE:
    from torch_mist.distributions.utils import conditional_transformed_normal, transformed_normal

    q_y_x = conditional_transformed_normal(
        input_dim=y_dim,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
        transform_name=conditional_transform_name,
        n_transforms=n_conditional_transforms,
    )

    q_y = transformed_normal(
        input_dim=y_dim,
        hidden_dims=hidden_dims,
        transform_name=marginal_transform_name,
        n_transforms=n_marginal_transforms,
    )

    return DoE(
        conditional_y_x=q_y_x,
        marginal_y=q_y,
    )
