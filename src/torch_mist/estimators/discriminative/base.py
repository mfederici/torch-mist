from abc import abstractmethod
from copy import deepcopy
from typing import Optional, Union

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch_mist.estimators.base import MutualInformationEstimator
from torch_mist.critic.base import Critic
from torch_mist.critic.separable import SeparableCritic
from torch_mist.utils.caching import cached, reset_cache_before_call, reset_cache_after_call


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

        # We sample from the proposal distribution (off diagonal elements)
        idx = torch.arange(max_samples * n_samples).to(self._samples.device).view(max_samples, n_samples).long()
        idx = (idx % n_samples + torch.div(idx, n_samples, rounding_mode='floor') + 1) % max_samples
        y_ = self._samples[idx.T]

        return y_

    def update(self):
        self._samples = None


class DiscriminativeMutualInformationEstimator(MutualInformationEstimator):
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
        assert f.ndim == y.ndim - 1
        return f

    def sample_proposal(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        # Sample from the proposal distribution r(y|x) [M',N, ..., Y_DIM] with M' as the number of mc_samples
        if isinstance(self.proposal, ConditionalDistribution):
            y_ = self.proposal.condition(x).sample(sample_shape=torch.Size([n_samples]))
        else:
            y_ = self.proposal.sample(sample_shape=torch.Size([n_samples]))
            # The shape of the samples from the proposal distribution is [M', N, ..., Y_DIM]
        assert y_.ndim == x.ndim+1 and y_.shape[0] == n_samples
        return y_

    @cached
    def critic_on_negatives(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_samples = self.mc_samples
        N = x.shape[0]

        if isinstance(self.proposal, EmpiricalDistribution):
            self.proposal.add_samples(y)

        # Negative MC values are interpreted as M' = N + mc_samples
        if n_samples <= 0:
            n_samples = N + n_samples

        # If we are sampling using the empirical distribution and the critic is separable
        # We can use an efficient implementation
        if isinstance(self.proposal, EmpiricalDistribution) and isinstance(self.critic, SeparableCritic):
            # Take the N stored samples from y (stored in _samples). Shape [N, 1, ...,Y_DIM]
            y_ = y[:N].unsqueeze(1)
            # Compute the critic on them. Shape [N, N, ...]
            f_ = self.critic(x, y_)
            if n_samples != N:
                # We keep only n_sample off-diagonal elements
                mask = torch.ones(N, N).to(x.device)
                mask = torch.triu(mask, 1) - torch.triu(mask, n_samples + 1) + torch.tril(mask, -N + n_samples)
                mask = mask.bool()
                while mask.ndim < f_.ndim:
                    mask = mask.unsqueeze(-1)
                mask = mask.expand_as(f_)
                f_ = torch.masked_select(f_, mask).reshape(N, n_samples).transpose(0, 1)
        else:
            # Sample from the proposal r(y|x) [M', N, ..., Y_DIM] with M' as the number of mc_samples
            y_ = self.sample_proposal(x, n_samples)
            assert y_.shape[0] == n_samples and y_.ndim == x.ndim+1

            # Compute the log-ratio on the samples from the proposal p(x)r(y|x) [M',N,...]
            f_ = self.critic(x, y_)

        assert f_.ndim == x.ndim, f"Expected {x.ndim} dimensions, got {f_.ndim}"
        assert f_.shape[0] == n_samples, f"Expected {n_samples} samples, got {f_.shape[0]}"

        if isinstance(self.proposal, EmpiricalDistribution):
            self.proposal.update()

        return f_

    def compute_log_ratio(self, x: torch.Tensor, y: torch.Tensor, f: torch.Tensor, f_: torch.tensor):
        raise NotImplementedError()

    @ cached
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.ndim == y.ndim
        # Compute the log-ratio p(y|x)/p(y) on samples from p(x)p(y|x).
        # The expected shape is [N]

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)p(y|x), with shape [N]
        f = self.critic_on_positives(x, y)

        # Evaluate the unnormalized_log_ratio f(x,y) on the samples from p(x)r(y|x), with shape [N, M']
        f_ = self.critic_on_negatives(x, y)

        log_ratio = self.compute_log_ratio(x, y, f, f_)
        assert log_ratio.ndim == y.ndim - 1

        return log_ratio

    @reset_cache_before_call
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.expected_log_ratio(x=x, y=y)

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  (critic): ' + str(self.critic).replace('\n', '\n' + '  ') + '\n'
        s += '  (mc_samples): ' + str(self.mc_samples) + '\n'
        s += ')'

        return s




