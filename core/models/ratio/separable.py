from typing import Optional, List

import torch
from pyro.nn import DenseNN

from torch import nn

from core.models.ratio.base import RatioEstimator


class SeparableRatioEstimator(RatioEstimator):
    def __init__(
            self,
            f_x: Optional[nn.Module] = None,
            f_y: Optional[nn.Module] = None,
    ):
        '''
        Model the critic as the product of two feature extractors
        f(x,y) = f_x(x)^T f_y(y)
        Note that the baseline can return a constant value (e.g. NWJ estimator)
        :param f_x: a (learnable) model returning a real vector of size D when given x. The identity is used if None is specified.
        :param f_y: a (learnable) model returning a real vector of size D when given y. The identity is used if None is specified.
        '''

        super(SeparableRatioEstimator, self).__init__()
        self.f_x = (lambda x: x) if f_x is None else f_x
        self.f_y = (lambda y: y) if f_y is None else f_y

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                ) -> torch.Tensor:

        if x.ndim == y.ndim:
            K = torch.einsum('acb, acb -> ac', self.f_x(x), self.f_y(y))
        else:
            K = torch.einsum('ab, acb -> ac', self.f_x(x), self.f_y(y))
        return K


class SeparableRatioEstimatorMLP(SeparableRatioEstimator):
    def __init__(self, x_dim: int, y_dim: int, hidden_dims: List[int], transform_x: bool = True, transform_y: bool = False, out_dim: Optional[int] = None):

        if not transform_y:
            out_dim = y_dim
        elif not transform_x:
            out_dim = x_dim
        else:
            assert out_dim is not None, "out_dim must be specified if both transform_x and transform_y are True"

        if transform_x:
            f_x = DenseNN(x_dim, hidden_dims, param_dims=[out_dim])
        else:
            f_x = None

        if transform_y:
            f_y = DenseNN(y_dim, hidden_dims, param_dims=[out_dim])
        else:
            f_y = None

        super(SeparableRatioEstimatorMLP, self).__init__(
            f_x=f_x,
            f_y=f_y,
        )