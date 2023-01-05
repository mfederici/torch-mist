import math
from functools import partial
from typing import Optional, List

import torch
from pyro.nn import DenseNN

from torch import nn, Tensor
import torch.nn.functional as F

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



class Projection(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int =2048,
                 out_dim: int = 128,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6)
                 ):
        super().__init__()
        self.out_dim = out_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            norm_layer(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=-1)


class SimCLRRatioEstimator(SeparableRatioEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dim: int = 1024,
            out_dim: int = 128,
            norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            temperature: float = 0.1,
    ):

        assert x_dim == y_dim
        projector = Projection(input_dim=x_dim, hidden_dim=hidden_dim, out_dim=out_dim, norm_layer=norm_layer)
        super(SimCLRRatioEstimator, self).__init__(
            f_x=projector,
            f_y=projector,
        )

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                ) -> torch.Tensor:

        # SimCLR can be used only when the negative samples are from the same batch
        assert (torch.equal(y[:, 0], y[:, 1]) and y.shape[1] == x.shape[0]) or y.shape[1] == 1

        y = y[:, 0]
        assert x.shape == y.shape

        hx = self.f_x(x)
        hy = self.f_y(y)

        # Adapted code from the pl_bolts repo (SimCLR)
        h = torch.cat([hx, hy], dim=0)
        K = torch.mm(h, h.t())
        sim = torch.exp(K / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=self.eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(hx * hy, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        return loss

