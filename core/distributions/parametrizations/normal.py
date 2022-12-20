import torch
from torch import nn
from torch.distributions import Normal, Distribution, Independent
from torch.nn.functional import softplus

from core.distributions.parametrizations import ParametrizedDistribution


class ParametrizedNormal(ParametrizedDistribution):
    def __init__(self,
                 loc=None,
                 scale=None,
                 optimize_loc=False,
                 optimize_scale=False,
                 epsilon=1e-6,
                 sigma_scale=1.0
                 ):

        if (loc is None) and (scale is None):
            n_params = 2
        else:
            n_params = 1

        super(ParametrizedNormal, self).__init__(n_params=n_params)
        self.epsilon = epsilon
        self.sigma_scale = sigma_scale

        if optimize_loc:
            self.loc = nn.Parameter(loc)
        elif not (loc is None):
            self.register_buffer('loc', loc)
        else:
            self.loc = None

        if scale is None:
            self.scale_log = None
        else:
            if not isinstance(scale, torch.Tensor):
                scale = torch.zeros(1)+scale
            scale_log = torch.log(torch.exp(scale) - 1.0)
            if optimize_scale:
                self.scale_log = nn.Parameter(scale_log)
            elif not (scale is None):
                self.register_buffer('scale_log', scale_log)

    def forward(self, x: torch.Tensor) -> Distribution:
        if self.loc is None and self.scale_log is None:
            loc, scale_log = torch.tensor_split(x, 2, dim=-1)
            loc = loc.squeeze(-1)
            scale_log = scale_log.squeeze(-1)

            # Corner case for 1d input
            if x.shape[-1] == 2:
                loc = loc.unsqueeze(-1)
                scale_log = scale_log.unsqueeze(-1)

        elif self.loc is None:
            loc = x
            scale_log = self.scale_log
        else:
            loc = self.loc
            scale_log = x

        scale = softplus(scale_log)*self.sigma_scale + self.epsilon
        dist = Normal(loc=loc, scale=scale)

        return Independent(dist, 1)
