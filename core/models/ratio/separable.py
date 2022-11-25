from typing import Optional

import torch

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
