from typing import Optional, Dict

import torch
from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from core.distributions.joint.base import JointDistribution
from core.models.mi_estimator.base import MutualInformationEstimator


class GenerativeMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            joint_xy: Optional[JointDistribution] = None,
            conditional_y_x: Optional[ConditionalDistribution] = None,
            conditional_a_y: Optional[ConditionalDistribution] = None,
            marginal_x: Optional[Distribution] = None,
            marginal_y: Optional[Distribution] = None,
            marginal_a: Optional[Distribution] = None,
            h_x: Optional[float] = None,
            h_y: Optional[float] = None,
            h_a: Optional[float] = None,

    ):
        super().__init__()

        n_models = 0
        if joint_xy is not None:
            n_models += 1
        if conditional_y_x is not None:
            n_models += 1
        if conditional_a_y is not None:
            n_models += 1
        assert n_models == 1, "Only one of joint_xy, conditional_y_x, conditional_a_y can be specified"

        # Using r_\theta(x,y)
        self.joint_xy = joint_xy

        # Using r_\theta(y|x)p(x)
        self.conditional_y_x = conditional_y_x

        # Using r_\theta(y|a)p(x)
        self.conditional_a_y = conditional_a_y

        # Saving the model for s(x), s(y) and s(a) (if provided) corresponding to the marginals
        self.marginal_x = marginal_x
        self.marginal_y = marginal_y
        self.marginal_a = marginal_a

        # Saving the entropy of the marginals (if provided)
        self.h_x = h_x
        self.h_y = h_y
        self.h_a = h_a

        self._cached_x = None
        self._cached_r_y_X = None

    def _compute_entropy(self, value: torch.Tensor, marginal: Optional[Distribution], h: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if marginal is not None and h is None:
            assert value.ndim >= 3
            value = value.view(-1, *value.shape[2:])

            # Compute the entropy
            h = - marginal.log_prob(value).mean()
        return h

    def compute_entropy_y(self, y: torch.Tensor) -> Optional[torch.Tensor]:
        return self._compute_entropy(y, self.marginal_y, self.h_y)

    def compute_entropy_x(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return self._compute_entropy(x, self.marginal_x, self.h_x)

    def compute_entropy_a(self, a: torch.Tensor) -> Optional[torch.Tensor]:
        return self._compute_entropy(a, self.marginal_a, self.h_a)

    def compute_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            a: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:

        # Compute the ratio using the primal bound
        estimates = {}

        if self.joint_xy is not None:
            # Compute E[log r(x,y)] - H(x) - H(y)

            if x.ndim == y.ndim-1:
                x = x.unsqueeze(1)

            x = x + y*0

            # Compute the log probability log p(X=x, Y=y)
            r_XY = self.joint_xy.log_prob({'x': x, 'y': y})

            estimates['grad'] = r_XY.mean()

            h_x = self.compute_entropy_x(x)
            h_y = self.compute_entropy_y(y)

            if h_x is not None and h_y is not None:
                estimates['h_x'] = h_x
                estimates['h_y'] = h_y
                estimates['value'] = estimates['grad'] + h_x + h_y
                estimates['grad'] = estimates['grad'] - h_x - h_y

        elif self.conditional_y_x is not None:
            # Compute E[log r(y|x)] - H(y)

            self._cached_x = x

            # Unsqueeze an empty dimension so that x and y have the same number of dimensions
            x = x.unsqueeze(1)
            assert y.ndim == x.ndim

            # Compute r(y|X=x) for the given x
            r_y_X = self.conditional_y_x.condition(x)

            # Evaluate the log probability log r(Y=y|X=x)
            log_r_Y_X = r_y_X.log_prob(y)

            self._cached_r_y_X = r_y_X

            estimates['grad'] = log_r_Y_X.mean()

            # The ratio of gradient is the same as the cross-entropy (for fixed entropy)
            h_y = self.compute_entropy_y(y)

            if h_y is not None:
                estimates['h_y'] = h_y
                estimates['value'] = estimates['grad'] + h_y
                estimates['grad'] = estimates['grad'] - h_y

        elif self.conditional_a_y is not None:
            # Compute E[log r(a|y)] - H(a) if a is provided
            if a is None:
                estimates['value'] = torch.zeros(1, device=x.device)
                estimates['grad'] = torch.zeros(1, device=x.device)
            else:
                r_a_Y = self.conditional_a_y.condition(y)

                if a.ndim == 1:
                    a = a.unsqueeze(-1)

                if a.ndim < y.ndim:
                    # Unsqueeze a to have the same shape as y
                    a = a.unsqueeze(1).repeat(1, y.shape[1], 1)

                # Compute the cross-entropy
                log_r_A_Y = r_a_Y.log_prob(a).mean(1)

                estimates['grad'] = log_r_A_Y.mean()

                h_a = self.compute_entropy_a(a)

                if h_a is not None:
                    estimates['h_a'] = h_a
                    estimates['value'] = estimates['grad'] + h_a
                    estimates['grad'] = estimates['grad'] - h_a
        else:
            raise Exception("No proposal provided")

        return estimates


class BA(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            h_y: Optional[float] = None,
    ):
        super().__init__(
            conditional_y_x=conditional_y_x,
            h_y=h_y,
        )


class DoE(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            marginal_y: Distribution,
    ):

        super().__init__(
            conditional_y_x=conditional_y_x,
            marginal_y=marginal_y,
        )


class GM(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            joint_xy: JointDistribution,
            marginal_y: Distribution,
            marginal_x: Distribution,
    ):

        super().__init__(
            joint_xy=joint_xy,
            marginal_y=marginal_y,
            marginal_x=marginal_x,
        )


class ABC(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            conditional_a_y: ConditionalDistribution,
            marginal_a: Optional[Distribution] = None,
            h_a: Optional[float] = None,

    ):

        super().__init__(
            conditional_a_y=conditional_a_y,
            marginal_a=marginal_a,
            h_a=h_a,
        )
