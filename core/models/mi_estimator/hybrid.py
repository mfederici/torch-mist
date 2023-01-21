from typing import Optional

import torch
from torch.distributions import Distribution

from core.distributions.joint.base import JointDistribution
from core.models.mi_estimator.base import MutualInformationEstimator
from core.models.mi_estimator.discriminative import DiscriminativeMutualInformationEstimator
from core.models.mi_estimator.generative import GenerativeMutualInformationEstimator


class HybridMutualInformationEstimator(MutualInformationEstimator):
    def __init__(
            self,
            generative_estimator: Optional[GenerativeMutualInformationEstimator]=None,
            discriminative_estimator: Optional[DiscriminativeMutualInformationEstimator]=None,
    ):
        super().__init__()
        self.generative_estimator = generative_estimator
        self.discriminative_estimator = discriminative_estimator

    @property
    def h_a(self):
        return self.generative_estimator.h_a

    @h_a.setter
    def h_a(self, value):
        self.generative_estimator.h_a = value

    def sample_proposal(self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor):
        # Determine the number of samples to draw from the proposal
        N = x.shape[0]
        n_samples = self.discriminative_estimator.neg_samples
        if n_samples <= 0:
            n_samples = N - n_samples

        # 1) If no generative estimator is provided, we assume that the proposal is the same as the product of the marginals
        if self.generative_estimator is None:
            x_, y_ = self.discriminative_estimator.sample_marginals(x, y)

        # 2) If we are using a, check that all the attributes in the same batch are the same and sample from the marginal
        elif self.generative_estimator.conditional_a_y is not None:
            if a is not None:
                same_a = ((a == a[0]).sum(-1) == a.shape[-1]).sum() == a.shape[0]
                assert same_a, 'All a must be the same within a batch'
            # This is equivalent to sampling from the p(x)p(y|a)
            x_, y_ = self.discriminative_estimator.sample_marginals(x, y)

        # 3) If we are using a conditional proposal r(y|x), sample from it
        elif self.generative_estimator.conditional_y_x is not None:
            assert torch.equal(x, self.generative_estimator._cached_x)
            r_y_X = self.generative_estimator._cached_r_y_X
            assert isinstance(r_y_X, Distribution)
            y_ = r_y_X.sample(torch.Size([n_samples])).squeeze(-2).permute(1, 0, 2)
            x_ = x

        # 4) Otherwise, if we are using a joint proposal r(x,y), sample from it
        else:
            r_XY = self.generative_estimator.joint_xy
            assert isinstance(r_XY, JointDistribution)
            sample = r_XY.sample(torch.Size([n_samples]))
            x_ = sample['x']
            y_ = sample['y']

        assert x_.shape[0] == N
        assert y_.shape[0] == N or y_.shape[0] == 1
        assert y_.shape[1] == n_samples

        return x_, y_

    def compute_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            x_: Optional[torch.Tensor] = None,
            y_: Optional[torch.Tensor] = None,
            a: Optional[torch.Tensor] = None,
    ):
        estimates = {}

        assert x_ is None, 'x_ must be None'
        assert y_ is None, 'y_ must be None'

        if a is not None:
            if a.ndim == 1:
                a = a.unsqueeze(-1)

        if self.generative_estimator is not None:
            generative_estimates = self.generative_estimator.compute_ratio(x, y, a=a)
        else:
            generative_estimates = {
                'value': torch.zeros(1).to(x.device).sum(),
                'grad': torch.zeros(1).to(x.device).sum()
            }

        for key, value in generative_estimates.items():
            estimates['gen/' + key] = value

        if self.discriminative_estimator is not None:
            # Sample the proposal r(x, y)
            x_, y_ = self.sample_proposal(x, y, a)
            # Compute the discriminative value
            discriminative_estimates = self.discriminative_estimator.compute_ratio(x, y, x_, y_)
        else:
            discriminative_estimates = {
                'value': torch.zeros(1).to(x.device).sum(),
                'grad': torch.zeros(1).to(x.device).sum()
            }

        for key, value in discriminative_estimates.items():
            estimates['dis/' + key] = value

        if 'value' in generative_estimates and 'value' in discriminative_estimates:
            estimates['value'] = generative_estimates['value'] + discriminative_estimates['value']
        if 'grad' in generative_estimates and 'grad' in discriminative_estimates:
            estimates['grad'] = generative_estimates['grad'] + discriminative_estimates['grad']

        return estimates
