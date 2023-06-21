import math
from abc import abstractmethod
from typing import Optional, List, Any

import torch
import torch.nn.functional as F
from pyro.distributions import ConditionalDistribution
from torch import nn

from torch_mist.estimators.base import MutualInformationEstimator, Estimation
from torch_mist.critic import Critic, SeparableCritic
from torch_mist.baselines import Baseline, BatchLogMeanExp, ConstantBaseline, ExponentialMovingAverage, \
    InterpolatedBaseline, LearnableJointBaseline, LearnableMLPBaseline, LearnableJointMLPBaseline
from torch_mist.critic.utils import critic


class DiscriminativeMutualInformationEstimator(MutualInformationEstimator):
    lower_bound = True
    upper_bound = False

    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            mc_samples: int = 1,
    ):
        super().__init__()
        self.unnormalized_log_ratio = unnormalized_log_ratio
        self.mc_samples = mc_samples
        self.proposal = None

    def sample_proposal(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_ = None
        N = y.shape[0]

        # For negative or zero values we consider N-self.n_samples instead
        if self.mc_samples <= 0:
            n_samples = N + self.mc_samples
        else:
            n_samples = self.mc_samples

        if self.proposal is None:
            # If we want to the whole batch as negatives and the unnormalized log-ratio is separable
            if (
                    self.mc_samples == 0 and
                    isinstance(self.unnormalized_log_ratio, SeparableCritic)
            ):
                # simply unsqueeze an empty dimension at the beginning since we don't want to forward (N**2) samples
                y_ = y[:, 0].unsqueeze(0)
            else:
                # Otherwise, we sample from the proposal distribution (off diagonal elements)
                # This indexing operation takes care of selecting the appropriate (off-diagonal) y
                idx = torch.arange(N * n_samples).to(y.device).view(N, n_samples).long()
                idx = (idx % n_samples + torch.div(idx, n_samples, rounding_mode='floor') + 1) % N
                y_ = y[:, 0][idx]
        else:
            # Conditional proposal
            if isinstance(self.proposal, ConditionalDistribution):
                proposal = self.proposal.condition(x)
            # Unconditional proposal
            else:
                proposal = self.proposal

            # sample from the proposal distribution
            y_ = proposal.sample([n_samples]).permute(1, 0, 2)

        return y_

    def log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Estimation:

        # Compute the log-ratio q(y|x)/p(y) on samples from p(x)p(y|x).
        # The expected shape is [N, M], with M as the number of samples from p(y|x)

        # Evaluate the unnormalized_log_ratio on the samples from p(x)p(y|x) [N, M]
        f = self.unnormalized_log_ratio(x, y)

        # Sample from the proposal distribution r(y|x) [N, M'] with M' as the number of mc_samples
        y_ = self.sample_proposal(x, y)

        # Compute the log-ratio on the samples from the proposal p(x)r(y|x) [N, M']
        f_ = self.unnormalized_log_ratio(x, y_)

        mi_value = self._compute_log_ratio(x=x, y=y, f=f, y_=y_, f_=f_)
        mi_grad = self._compute_log_ratio_grad(x=x, y=y, f=f, y_=y_, f_=f_)

        if mi_grad is None:
            mi_grad = -mi_value

        return Estimation(value=mi_value, loss=mi_grad)

    @abstractmethod
    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _compute_log_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  (ratio_estimator): ' + str(self.unnormalized_log_ratio).replace('\n', '\n' + '  ') + '\n'
        s += '  (mc_samples): ' + str(self.mc_samples) + '\n'
        s += ')'

        return s


class NWJ(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            mc_samples: int = 1,
    ):
        super().__init__(
            unnormalized_log_ratio=unnormalized_log_ratio,
            mc_samples=mc_samples,
        )

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        log_ratio = f - f_.exp().mean(1).unsqueeze(1) + 1

        return log_ratio


class TUBA(NWJ):
    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            baseline: Baseline,
            grad_baseline: Optional[Baseline] = None,
            mc_samples: int = 1
    ):

        super().__init__(
            unnormalized_log_ratio=unnormalized_log_ratio,
            mc_samples=mc_samples,
        )

        self.baseline = baseline
        self.grad_baseline = grad_baseline

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:

        value_b = self.baseline.forward(f_, x, y)
        if value_b.ndim == 1:
            value_b = value_b.unsqueeze(1)
        assert value_b.ndim == f_.ndim

        return super()._compute_log_ratio(
            x=x,
            y=y,
            f=f - value_b,
            y_=y_,
            f_=f_ - value_b
        )

    def _compute_log_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> Optional[torch.Tensor]:

        if self.grad_baseline is None:
            return None

        grad_b = self.grad_baseline.forward(f_, x, y)
        if grad_b.ndim == 1:
            grad_b = grad_b.unsqueeze(1)
        assert grad_b.ndim == f_.ndim

        return super()._compute_log_ratio_grad(
            x=x,
            y=y,
            f=f - grad_b,
            y_=y_,
            f_=f_ - grad_b
        )

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  (ratio_estimator): ' + str(self.unnormalized_log_ratio).replace('\n', '\n' + '  ') + '\n'
        s += '  (baseline): ' + str(self.baseline).replace('\n', '\n' + '  ') + '\n'
        if self.grad_baseline is not None:
            s += '  (grad_baseline): ' + str(self.grad_baseline).replace('\n', '\n' + '  ') + '\n'
        s += '  (mc_samples): ' + str(self.mc_samples) + '\n'
        s += ')'

        return s


class MINE(TUBA):
    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            mc_samples: int = 1,
            gamma: float = 0.9,
    ):
        super().__init__(
            unnormalized_log_ratio=unnormalized_log_ratio,
            mc_samples=mc_samples,
            baseline=BatchLogMeanExp(dim=2),
            grad_baseline=ExponentialMovingAverage(gamma=gamma),
        )


class InfoNCE(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            unnormalized_log_ratio: SeparableCritic,
    ):
        # Note that this can be equivalently obtained by extending TUBA with a BatchLogMeanExp(dim=1) baseline
        # This implementation saves some computation
        super().__init__(
            unnormalized_log_ratio=unnormalized_log_ratio,
            mc_samples=0,  # 0 signifies the whole batch is used as negative samples
        )

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: torch.Tensor,
            f: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        N, M = f_.shape[0], f_.shape[1]

        # Compute the estimation for the normalization constant
        # log 1/M \sum_{j=1}^M f_[i,j] = logsumexp(f_,1).mean(0) - log M
        log_Z_value = (torch.logsumexp(f_, 1) - math.log(M)).unsqueeze(1)

        log_ratio = f - log_Z_value

        return log_ratio


class JS(NWJ):
    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            mc_samples: int = 1,
    ):
        super().__init__(
            unnormalized_log_ratio=unnormalized_log_ratio,
            mc_samples=mc_samples,
        )

    def _compute_log_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> Optional[torch.Tensor]:
        ratio_grad = F.softplus(-f).mean(1) + F.softplus(f_).mean(1)
        return ratio_grad


class AlphaTUBA(TUBA):
    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            baseline: Baseline = ConstantBaseline(1.0),
            alpha: float = 0.5,
            mc_samples: int = -1,
    ):
        alpha_baseline = InterpolatedBaseline(
            baseline_1=BatchLogMeanExp(dim=1),
            baseline_2=baseline,
            alpha=alpha
        )

        super().__init__(
            unnormalized_log_ratio=unnormalized_log_ratio,
            baseline=alpha_baseline,
            mc_samples=mc_samples,
        )


class SMILE(MINE, JS):
    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            mc_samples: int = 1,
            tau: float = 5.0,
    ):
        MINE.__init__(
            self,
            unnormalized_log_ratio=unnormalized_log_ratio,
            mc_samples=mc_samples,
        )
        self.tau = tau

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> torch.Tensor:
        return MINE._compute_log_ratio(
            self,
            x=x, y=y, f=f, y_=y_,
            f_=torch.clamp(f_, -self.tau, self.tau)
        )

    def _compute_log_ratio_grad(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return JS._compute_log_ratio_grad(self, x=x, y=y, f=f, y_=y_, f_=f_)


class FLO(DiscriminativeMutualInformationEstimator):
    def __init__(
            self,
            unnormalized_log_ratio: Critic,
            baseline: LearnableJointBaseline,
            mc_samples: int = 1,
    ):
        super().__init__(
            unnormalized_log_ratio=unnormalized_log_ratio,
            mc_samples=mc_samples,
        )
        self.baseline = baseline

    def _compute_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            f: torch.Tensor,
            y_: torch.Tensor,
            f_: torch.Tensor,
    ) -> torch.Tensor:
        b = self.baseline(f_, x, y)

        if b.ndim == 1:
            b = b.unsqueeze(1)
        assert b.ndim == f_.ndim

        M = f_.shape[1]

        log_ratio = -(b + (torch.logsumexp(f_, 1).unsqueeze(1) - f - b).exp() / M) + 1

        return log_ratio


# Factory functions

def nwj(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        mc_samples=1,
        critic_type='joint',
        **kwargs
) -> NWJ:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        **kwargs

    )

    return NWJ(
        unnormalized_log_ratio=url_nn,
        mc_samples=mc_samples,
    )


def tuba(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='joint',
        mc_samples=1,
        nonlinearity: Any = nn.ReLU(True),
        **kwargs
) -> TUBA:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        nonlinearity=nonlinearity,
        **kwargs
    )
    baseline = LearnableMLPBaseline(
        x_dim=x_dim,
        hidden_dims=hidden_dims,
        nonlinearity=nonlinearity,
    )

    return TUBA(
        unnormalized_log_ratio=url_nn,
        baseline=baseline,
        mc_samples=mc_samples,
    )


def mine(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='joint',
        mc_samples=1,
        gamma: float = 0.9,
        **kwargs
) -> MINE:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        **kwargs
    )

    return MINE(
        unnormalized_log_ratio=url_nn,
        mc_samples=mc_samples,
        gamma=gamma,
    )


def infonce(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='separable',
        **kwargs
) -> InfoNCE:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        **kwargs
    )

    return InfoNCE(
        unnormalized_log_ratio=url_nn,
    )


def js(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='joint',
        mc_samples=1,
        **kwargs
) -> JS:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        **kwargs
    )

    return JS(
        unnormalized_log_ratio=url_nn,
        mc_samples=mc_samples,
    )


def alpha_tuba(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='joint',
        alpha: float = 0.01,
        nonlinearity: Any = nn.ReLU(True),
        learnable_baseline: bool = True,
        mc_samples=-1,
        **kwargs
) -> AlphaTUBA:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        nonlinearity=nonlinearity,
        **kwargs
    )
    if learnable_baseline:
        baseline = LearnableMLPBaseline(
            x_dim=x_dim,
            hidden_dims=hidden_dims,
            nonlinearity=nonlinearity,
        )
    else:
        baseline = ConstantBaseline(1.0)

    return AlphaTUBA(
        unnormalized_log_ratio=url_nn,
        baseline=baseline,
        alpha=alpha,
        mc_samples=mc_samples,
    )


def smile(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='joint',
        mc_samples=1,
        tau: float = 5.0,
        **kwargs
) -> SMILE:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        **kwargs
    )

    return SMILE(
        unnormalized_log_ratio=url_nn,
        mc_samples=mc_samples,
        tau=tau,
    )


def flo(
        x_dim: int,
        y_dim: int,
        hidden_dims: List[int],
        critic_type='joint',
        mc_samples=1,
        nonlinearity: Any = nn.ReLU(True),
        **kwargs
) -> FLO:
    url_nn = critic(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        critic_type=critic_type,
        nonlinearity=nonlinearity,
        **kwargs
    )
    baseline = LearnableJointMLPBaseline(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        nonlinearity=nonlinearity,
    )

    return FLO(
        unnormalized_log_ratio=url_nn,
        baseline=baseline,
        mc_samples=mc_samples,
    )
