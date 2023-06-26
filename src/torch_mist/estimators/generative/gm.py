from typing import Optional, List, Union

import torch
from torch.distributions import Distribution

from torch_mist.estimators.generative.base import GenerativeMutualInformationEstimator


class GM(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            joint_xy: Distribution,
            marginal_y: Optional[Distribution] = None,
            marginal_x: Optional[Distribution] = None,
            entropy_y: Optional[torch.Tensor] = None,
            entropy_x: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.joint_xy = joint_xy
        assert (marginal_y is None) ^ (
                entropy_y is None), 'Either the marginal distribution or the marginal entropy must be provided'
        assert (marginal_x is None) ^ (
                entropy_x is None), 'Either the marginal distribution or the marginal entropy must be provided'
        self.marginal_y = marginal_y
        self.entropy_y = entropy_y
        self.marginal_x = marginal_x
        self.entropy_x = entropy_x

        self._cached_x = None
        self._cached_y = None
        self._cached_entropy_xy = None
        self._cached_entropy_x = None

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.entropy_y is not None:
            return -torch.FloatTensor(self.entropy_y).unsqueeze(0).unsqueeze(1).to(y.device)
        else:
            return self.marginal_y.log_prob(y)

    def log_prob_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.entropy_x is not None:
            return -torch.FloatTensor(self.entropy_x).unsqueeze(0).unsqueeze(1).to(x.device)
        else:
            log_p_x = self.marginal_x.log_prob(x)
            if x.ndim == 2:
                log_p_x = log_p_x.unsqueeze(1)
            return log_p_x

    def log_prob_y_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._cached_x = x
        self._cached_y = y

        # Compute E[-log r(y|x)]
        x = x + y * 0
        y = y + x * 0

        xy = torch.cat([x, y], dim=-1)

        log_r_XY = self.joint_xy.log_prob(xy)
        log_r_X = self.log_prob_x(x, y)
        log_r_Y_X = log_r_XY - log_r_X

        # Cache the entropy and the inputs x, y
        self._cached_entropy_xy = -log_r_XY.mean()
        self._cached_entropy_x = -log_r_X.mean()

        return log_r_Y_X

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ):
        assert torch.equal(x, self._cached_x), 'The input x is not the same as the cached input x'
        assert torch.equal(y, self._cached_y), 'The input y is not the same as the cached input y'

        entropy_xy = self._cached_entropy_xy
        entropy_x = self._cached_entropy_x
        entropy_y = -log_p_y.mean()

        # Optimize using maximum likelihood
        return entropy_y + entropy_x + entropy_xy

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(joint_yx): ' + str(self.joint_xy).replace('\n', '  \n') + '\n'
        if self.marginal_x is not None:
            s += '  ' + '(marginal_x): ' + str(self.marginal_x).replace('\n', '  \n') + '\n'
        if self.marginal_y is not None:
            s += '  ' + '(marginal_y): ' + str(self.marginal_y).replace('\n', '  \n') + '\n'
        s += ')' + '\n'
        return s


def gm(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        entropy_x: Optional[Union[torch.Tensor, float]] = None,
        entropy_y: Optional[Union[torch.Tensor, float]] = None,
        joint_transform_name: str = 'conditional_linear',
        n_joint_transforms: int = 1,
        marginal_transform_name: str = 'linear',
        n_marginal_transforms: int = 1,
) -> GM:
    from torch_mist.distributions.utils import transformed_normal

    joint_xy = transformed_normal(
        input_dim=x_dim + y_dim,
        hidden_dims=hidden_dims,
        transform_name=joint_transform_name,
        n_transforms=n_joint_transforms,
    )

    if entropy_x is None:
        marginal_x = transformed_normal(
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            transform_name=marginal_transform_name,
            n_transforms=n_marginal_transforms,
        )
    else:
        marginal_x = None
        if isinstance(entropy_x, float):
            entropy_x = torch.FloatTensor([entropy_x])
        entropy_x = entropy_x.squeeze()

    if entropy_y is None:
        marginal_y = transformed_normal(
            input_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name=marginal_transform_name,
            n_transforms=n_marginal_transforms,
        )
    else:
        marginal_y = None
        if isinstance(entropy_y, float):
            entropy_y = torch.FloatTensor([entropy_y])
        entropy_y = entropy_y.squeeze()

    return GM(
        joint_xy=joint_xy,
        marginal_x=marginal_x,
        marginal_y=marginal_y,
        entropy_x=entropy_x,
        entropy_y=entropy_y,
    )
