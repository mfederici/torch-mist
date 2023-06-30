from abc import abstractmethod
from typing import Optional, Union

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch_mist.estimators.base import MutualInformationEstimator
from torch_mist.critic import Critic, SeparableCritic
from torch_mist.utils.caching import cached, reset_cache_before_call


class EmpiricalDistribution(Distribution):
    def __init__(self):
        super().__init__(validate_args=False)
        self._samples = None

    def add_samples(self, samples):
        self._samples = samples

    def sample(self, sample_shape=torch.Size()):
        assert self._samples is not None
        assert len(sample_shape) == 1
        n_samples = sample_shape[0]
        max_samples = self._samples.shape[0]

        # Otherwise, we sample from the proposal distribution (off diagonal elements)
        idx = torch.arange(max_samples * n_samples).to(self._samples.device).view(max_samples, n_samples).long()
        idx = (idx % n_samples + torch.div(idx, n_samples, rounding_mode='floor') + 1) % max_samples
        y_ = self._samples[:, 0][idx].permute(1, 0, 2)

        return y_

    def update(self):
        pass


class DiscriminativeMutualInformationEstimator(MutualInformationEstimator)
    def __init__(
            self,
            critic: Critic,
            mc_samples: int = 1,
            proposal: Optional[Union[Distribution, ConditionalDistribution]] = None,
    ):
        super().__init__()
        self.critic = critic
        self.mc_samples = mc_samples
        if proposal is None:
            proposal = EmpiricalDistribution()
        self.proposal = proposal

    def unnormalized_log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.critic_on_positives(x, y)

    @cached
    def critic_on_positives(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f = self.critic(x, y)
        assert f.shape == y.shape[:-1]
        return f

    def sample_proposal(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        # Sample from the proposal distribution r(y|x) [N, M'] with M' as the number of mc_samples
        if isinstance(self.proposal, ConditionalDistribution):
            y_ = self.proposal.condition(x).sample(sample_shape=torch.Size([n_samples]))
            y_ = y_.permute(1, 0, 2)
            assert y_.shape[:2] == torch.Size([x.shape[0], n_samples])
        else:
            y_ = self.proposal.sample(sample_shape=torch.Size([n_samples]))
            # The shape of the samples from the proposal distribution is [M', Y_DIM]
            assert y_.ndim == 2 and y_.shape[0] == n_samples
            # We need to add a batch dimension to the samples from the proposal distribution
            y_ = y_.unsqueeze(0)
        return y_

    @cached
    def negative_critic(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = self.mc_samples
        # Negative MC values are interpreted as N + mc_samples
        if n_samples <= 0:
            n_samples = x.shape[0] + n_samples

        # Sample from the proposal r(y|x) [N, M', Y_DIM] with M' as the number of mc_samples
        y_ = self.sample_proposal(x, n_samples)
        assert y_.shape[1] == n_samples and y_.ndim==3

        # Compute the log-ratio on the samples from the proposal p(x)r(y|x) [N, M']
        f_ = self.critic(x, y_)
        assert f_.shape == y_.shape[:2]

        return f_

    def log_normalization_constant(self, x: torch.Tensor):


    @reset_cache_before_call
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert y.ndim == 2
        assert x.ndim == 2

        if isinstance(self.proposal, EmpiricalDistribution):
            self.proposal.add_samples(y)

        # Compute the log-ratio p(y|x)/p(y) on samples from p(x)p(y|x).
        # The expected shape is [N]

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)p(y|x), with shape [N]
        f = self.critic_on_positives(x, y)
        assert f.shape == y.shape[:-1]


        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)r(y|x), with shape [N, M']
        f_ = self.negative_critic(x)
        assert log_norm.shape == x.shape[:-1]


        if isinstance(self.proposal, EmpiricalDistribution):
            self.proposal.update()


        return u_log_ratio - log_norm.unsqueeze(-1)


    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  (critic): ' + str(self.critic).replace('\n', '\n' + '  ') + '\n'
        s += '  (mc_samples): ' + str(self.mc_samples) + '\n'
        s += ')'

        return s
