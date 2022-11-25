from typing import Optional, Tuple, List, Any

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

import pyro.distributions.transforms as pyro_transforms_module
import core.distributions.transforms as transforms_module
from core.distributions import conditional_transformed_normal

from core.models.mi_estimators.base import MutualInformationEstimator


class PrimalMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            r_y_x: ConditionalDistribution,
            p_y: Optional[Distribution] = None,
            h_y: Optional[float] = None
    ):
        '''
        Barber-Agakov lower bound on mutual information
        :param r_y_x: a conditional distribution r(y|x) over y for a given x
        :param p_y: a distribution for p(y)
        :param h_y: the entropy H(y)
        '''
        MutualInformationEstimator.__init__(self)
        self.r_y_x = r_y_x
        assert p_y is None or h_y is None
        self.p_y = p_y
        self.h_y = h_y

    def compute_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x = x.unsqueeze(1)

        assert y.ndim == x.ndim

        # Compute r(x|Y=y) for the given y
        r_y_X = self.r_y_x.condition(x)

        # Evaluate the log probability log r(X=x|Y=y)
        log_r_Y_X = r_y_X.log_prob(y)

        log_r_Y_X = log_r_Y_X

        # The mutual information gradient is the same as the cross-entropy
        mi_grad = log_r_Y_X

        h_y = None
        if self.p_y is not None:
            # Compute the entropy y
            h_y = -self.p_y.log_prob(y).mean()
        elif self.h_y is not None:
            h_y = self.h_y

        if h_y is not None:
            # The mutual information estimator is given by the difference between entropy and cross entropy
            mi_value = log_r_Y_X + h_y
        else:
            mi_value = None

        return mi_value, mi_grad

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += '\n  (r_y_x): ' + self.r_y_x.__repr__()
        if self.p_y is not None:
            s += '\n  (p_y): ' + self.p_y.__repr__()
        s += '\n)'
        return s


def ba(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        transform_name: str = "conditional_linear",
        n_transforms: int = 1,
        p_x: Optional[Distribution] = None,
        h_x: Optional[float] =None,
        **transform_params
):
    dist = conditional_transformed_normal(
        input_dim=x_dim,
        context_dim=y_dim,
        hidden_dims=hidden_dims,
        transform_name=transform_name,
        n_transforms=n_transforms,
        **transform_params
    )

    return PrimalMutualInformationEstimator(r_y_x=dist, p_x=p_x, h_x=h_x)

