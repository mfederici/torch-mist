from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution

from core.models.mi_estimator.base import MutualInformationEstimator
from core.models.ratio import SeparableRatioEstimator


class Projection(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 2048,
                 out_dim: int = 128,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
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


class SimCLR(MutualInformationEstimator):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            out_dim: int = 128,
            hidden_dim: int = 1024,
            norm_layer: Optional[nn.Module] = None,
            temperature: float = 0.1,
            predictor: Optional[nn.Module] = None,
            p_a: Optional[Distribution] = None,
            h_a: Optional[float] = None,
    ):
        assert x_dim == y_dim, "x_dim and y_dim must be equal"

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        projector = Projection(
            input_dim=x_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            norm_layer=norm_layer,
        )
        self.temperature = temperature

        MutualInformationEstimator.__init__(
            self,
            ratio_estimator=SeparableRatioEstimator(f_x=projector, f_y=projector),
            baseline=None,
            neg_samples=0,
            predictor=predictor,
            p_a=p_a,
            h_a=h_a,
        )

    def _compute_dual_ratio_value(self, x, y, f, f_, baseline):
        f = f / self.temperature

        f_xy = f_ / self.temperature
        f_yx = f_.T / self.temperature

        # Contrast x against itself
        f_xx = self.ratio_estimator(x, x.unsqueeze(0)) / self.temperature
        # Remove the diagonal
        f_xx = f_xx.tril(-1)[:, :-1] + f_xx.triu(1)[:, 1:]

        # Same for y
        f_yy = self.ratio_estimator(y.permute(1, 0, 2), y) / self.temperature
        f_yy = f_yy.tril(-1)[:, :-1] + f_yy.triu(1)[:, 1:]

        # Compute the log-noramlization constant
        log_Z_x = torch.logsumexp(
            torch.cat([f_xx, f_xy], 1), 1
        ).unsqueeze(1) - np.log(f_.shape[1]+f_xx.shape[1])

        log_Z_y = torch.logsumexp(
            torch.cat([f_yx, f_yy], 1), 1
        ).unsqueeze(1) - np.log(f_.shape[1]+f_yy.shape[1])

        log_Z = log_Z_x / 2.0 + log_Z_y / 2.0

        ratio_value = f - log_Z

        return ratio_value.mean()
