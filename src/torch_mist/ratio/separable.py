from typing import Optional, List

import torch
from pyro.nn import DenseNN

from torch import nn

from src.torch_mist.models.ratio.base import UnnormalizedRatioEstimator


class SeparableUnnormalizedRatioEstimator(UnnormalizedRatioEstimator):
    def __init__(
            self,
            f_x: Optional[nn.Module] = None,
            f_y: Optional[nn.Module] = None,
            temperature: float = 1.0
    ):
        '''
        Model the critic as the product of two feature extractors
        f(x,y) = f_x(x)^T f_y(y)
        Note that the baseline can return a constant value (e.g. NWJ estimator)
        :param f_x: a (learnable) model returning a real vector of size D when given x. The identity is used if None is specified.
        :param f_y: a (learnable) model returning a real vector of size D when given y. The identity is used if None is specified.
        '''

        super().__init__()
        self.f_x = (lambda x: x) if f_x is None else f_x
        self.f_y = (lambda y: y) if f_y is None else f_y
        self.temperature = temperature

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                ) -> torch.Tensor:

        f_x = self.f_x(x)
        f_y = self.f_y(y)

        if f_x.ndim < f_y.ndim:
            f_x = f_x.unsqueeze(1)

        # hack to expand to the same shape without specifying the number of repeats
        f_x = f_x + f_y * 0
        f_y = f_y + f_x * 0

        K = torch.einsum('acb, acb -> ac', f_x, f_y)
        return K/self.temperature


class SeparableRatioEstimatorMLP(SeparableUnnormalizedRatioEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dims: List[int],
            transform_x: bool = True,
            transform_y: bool = False,
            same_transform: bool = True,
            normalize_outputs: bool = False,
            out_dim: Optional[int] = None,
            temperature: float = 1.0
    ):

        if not transform_y:
            out_dim = y_dim
        elif not transform_x:
            out_dim = x_dim
        else:
            assert out_dim is not None, "out_dim must be specified if both transform_x and transform_y are True"

        assert not same_transform or not transform_y, "same_transform and transform_y cannot both be True"

        if transform_x:
            f_x = DenseNN(x_dim, hidden_dims, param_dims=[out_dim])
            if normalize_outputs:
                f_x = nn.Sequential(
                    f_x,
                    nn.Softmax(-1)
                )
        else:
            f_x = None

        if transform_y:
            f_y = DenseNN(y_dim, hidden_dims, param_dims=[out_dim])
            if normalize_outputs:
                f_y = nn.Sequential(
                    f_y,
                    nn.Softmax(-1)
                )
        elif same_transform:
            f_y = f_x
        else:
            f_y = None

        super(SeparableRatioEstimatorMLP, self).__init__(
            f_x=f_x,
            f_y=f_y,
            temperature=temperature
        )
