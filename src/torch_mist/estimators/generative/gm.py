from typing import List, Optional

import torch
from torch.distributions import Distribution

from torch_mist.distributions.joint import JointDistribution, ConditionalRatioDistribution
from torch_mist.estimators.generative.doe import DoE
from torch_mist.utils.caching import cached, reset_cache_after_call, reset_cache_before_call


class GM(DoE):
    def __init__(
            self,
            q_XY: JointDistribution,
            q_Y: Distribution,
            q_X: Distribution,
    ):
        q_Y_given_X = ConditionalRatioDistribution(q_XY, q_X)

        super().__init__(q_Y=q_Y, q_Y_given_X=q_Y_given_X)

        self.q_XY = q_XY
        self.q_Y = q_Y
        self.q_X = q_X
        self.q_X_given_Y = ConditionalRatioDistribution(q_XY, q_Y)

    @cached
    def approx_log_p_x(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        log_q_x = self.q_X.log_prob(x)
        assert log_q_x.shape == x.shape[:-1]
        # The shape is [N]
        return log_q_x

    @cached
    def approx_log_p_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The shape is [...]
        log_q_xy = self.q_XY.log_prob(x=x, y=y)
        return log_q_xy

    def approx_log_p_x_given_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_q_x_y = self.q_X_given_Y.condition(y=y).log_prob(x=x)
        return log_q_x_y

    @reset_cache_before_call
    def loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        log_q_xy = self.approx_log_p_xy(x=x, y=y)
        log_q_y = self.approx_log_p_y(y=y)
        log_q_x = self.approx_log_p_x(x=x)

        loss = -log_q_xy - log_q_y - log_q_x
        assert loss.ndim == y.ndim-1

        return loss.mean()

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  (q_XY): ' + str(self.q_XY).replace('\n', '\n  ') + '\n'
        if self.q_X is not None:
            s += '  (q_X): ' + str(self.q_X).replace('\n', '\n  ') + '\n'
        if self.q_Y is not None:
            s += '  (q_Y): ' + str(self.q_Y).replace('\n', '\n  ') + '\n'
        s += ')' + '\n'
        return s


def gm(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        joint_transform_name: str = 'conditional_linear',
        n_joint_transforms: int = 1,
        marginal_transform_name: str = 'linear',
        n_marginal_transforms: int = 1,
) -> GM:
    from torch_mist.distributions.utils import transformed_normal

    joint_xy = JointDistribution(
        joint_dist=transformed_normal(
            input_dim=x_dim + y_dim,
            hidden_dims=hidden_dims,
            transform_name=joint_transform_name,
            n_transforms=n_joint_transforms,
        ),
        dims=[x_dim, y_dim],
        names=['x', 'y'],
    )

    marginal_x = transformed_normal(
        input_dim=x_dim,
        hidden_dims=hidden_dims,
        transform_name=marginal_transform_name,
        n_transforms=n_marginal_transforms,
    )

    marginal_y = transformed_normal(
        input_dim=y_dim,
        hidden_dims=hidden_dims,
        transform_name=marginal_transform_name,
        n_transforms=n_marginal_transforms,
    )

    return GM(
        q_XY=joint_xy,
        q_X=marginal_x,
        q_Y=marginal_y,
    )
