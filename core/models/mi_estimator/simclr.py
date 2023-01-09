import math
from functools import partial
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.models.mi_estimator.base import MutualInformationEstimator


class Projection(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 out_dim: int = 128,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 ):
        super().__init__()
        self.out_dim = out_dim

        hidden_dims = [input_dim] + hidden_dims

        self.model = nn.Sequential()
        for i in range(len(hidden_dims) - 1):
            self.model.add_module(
                f"linear_{i}",
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
            )
            self.model.add_module(
                f"norm_{i}",
                norm_layer(hidden_dims[i + 1]),
            )
            self.model.add_module(
                f"relu_{i}",
                nn.ReLU(),
            )

        self.model.add_module(
            "linear_out",
            nn.Linear(hidden_dims[-1], self.out_dim, bias=False),
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
            hidden_dims: Optional[List[int]] = None,
            norm_layer: Optional[nn.Module] = None,
            temperature: float = 0.1,
            **kwargs
    ):
        assert x_dim == y_dim, "x_dim and y_dim must be equal"

        if hidden_dims is None:
            hidden_dims = [1024]

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        projector = Projection(
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            norm_layer=norm_layer,
        )
        self.temperature = temperature

        MutualInformationEstimator.__init__(
            self,
            # ratio_estimator=SeparableRatioEstimator(f_x=projector, f_y=projector),
            baseline=None,
            neg_samples=0,
            **kwargs
        )

        self.projector = projector

    def compute_dual_ratio(self, x: torch.Tensor, y: torch.Tensor, y_: Optional[torch.Tensor]=None) -> Tuple[
        Optional[torch.Tensor], torch.Tensor]:
        h_x = self.projector(x)
        h_y = self.projector(y).squeeze(1)

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([h_x, h_y], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out.t())

        # Remove the diagonal
        cov = cov.tril(-1)[:, :-1] + cov.triu(1)[:, 1:]
        neg = torch.logsumexp(cov / self.temperature, dim=-1) - math.log(cov.shape[-1])

        # Positive similarity
        pos = torch.sum(h_x * h_y, dim=-1) / self.temperature

        # The loss is given by (h(x).T h(y)/t).mean() - (logmeanexp([h(x), h(y)].T [h(x), h(y)]/t)).mean()
        # Analogously to InfoNCE
        loss = pos.mean() - neg.mean()

        # Gradient and loss value are the same
        return loss, loss

