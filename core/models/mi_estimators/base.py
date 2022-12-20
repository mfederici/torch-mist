from abc import abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.distributions import Distribution
from pyro.distributions import ConditionalDistribution

from core.models.baselines import Baseline, BatchLogMeanExp
from core.models.ratio import RatioEstimator

# The implementations in this work are generally based on
# 1) "On Variational Lower bounds of mutual information" https://arxiv.org/pdf/1905.06922.pdf
# 2) "Undertanding the Limitations of Variational Mutual Information Estimators https://arxiv.org/abs/1910.06222


class MutualInformationEstimator(nn.Module):
    def __init__(
            self,
            proposal: Optional[ConditionalDistribution] = None,
            predictive: Optional[ConditionalDistribution] = None,
            ratio_estimator: Optional[RatioEstimator] = None,
            baseline: Optional[Baseline] = None,
            grad_baseline: Optional[Baseline] = None,
            p_y: Optional[Distribution] = None,
            p_a: Optional[Distribution] = None,
            h_y: Optional[float] = None,
            h_a: Optional[float] = None,
            n_samples: int = 1,
            sample_gradient: bool = False
    ):
        super(MutualInformationEstimator, self).__init__()
        self.proposal = proposal
        self.predictive = predictive
        self.ratio_estimator = ratio_estimator
        self.baseline = baseline
        self.grad_baseline = grad_baseline
        self.p_y = p_y
        self.p_a = p_a
        self.h_y = h_y
        self.h_a = h_a
        self.n_samples = n_samples
        self.sample_gradient = sample_gradient

    def sample_proposal(self, x, y) -> torch.Tensor:
        N = y.shape[0]

        # For negative or zero values we consider N-self.n_samples instead
        if self.n_samples <= 0:
            n_samples = N - self.n_samples
        else:
            n_samples = self.n_samples

        # By default, we use the proposal is p(y)
        if self.proposal is None:
            # take other ys in the batch as samples from the marginal

            # This indexing operation takes care of selecting the appropriate (off-diagonal) y
            idx = torch.arange(N * n_samples).to(y.device).view(N, n_samples).long()
            idx = (idx % n_samples + torch.div(idx, n_samples, rounding_mode='floor') + 1) % N
            y_ = y[:, 0][idx]

        # Otherwise if we are given a proposal distribution we sample from it
        elif isinstance(self.proposal, ConditionalDistribution):
            r_y_X = self.proposal.condition(x)
            if self.sample_gradient:
                y_ = r_y_X.rsample([n_samples])
            else:
                y_ = r_y_X.sample([n_samples])

            y_ = y_.transpose(0, 1)
        else:
            raise NotImplementedError()

        # The output has shape [N, M, D]
        assert y_.ndim == y.ndim

        return y_

    def compute_primal_ratio(self, x: torch.Tensor, y: torch.Tensor, a: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # Computation of gradient and value of E_{p(x,y)}[log r(y|x)/p(y)]
        if self.proposal is None:
            if self.predictive is None or a is None:
                # By default we consider r(y|x) = p(y), therefore E_{p(x,y)}[log r(y|x)/p(y)] = 0
                ratio_value, ratio_grad = torch.zeros(1).to(x.device), torch.zeros(1).to(x.device)
            else:
                # Compute E_{p(x,y,a)}{log q(a|y)) + H(a)
                q_a_Y = self.predictive.condition(y)
                if a.ndim<y.ndim:
                    # Unsqueeze a to have the same shape as y
                    a_ = a.unsqueeze(1)+y*0
                else:
                    a_ = a
                # Compute the cross-entropy
                log_q_A_Y = q_a_Y.log_prob(a_).mean(1).mean(0)
                ratio_grad = log_q_A_Y
                h_a = None
                if self.p_a is not None:
                    h_a = - self.p_a.log_prob(a).mean(0)
                if self.h_a is not None:
                    h_a = self.h_a

                if h_a is not None:
                    ratio_value = log_q_A_Y + h_a
        else:
            # Unsqueeze an empty dimension so that x and y have the same number of dimensions
            x = x.unsqueeze(1)
            assert y.ndim == x.ndim

            # Compute r(x|Y=y) for the given y
            r_y_X = self.proposal.condition(x)

            # # Corner case for 1d
            # if y.shape[-1] == y:
            #     y = y.squeeze(-1)

            # Evaluate the log probability log r(X=x|Y=y)
            log_r_Y_X = r_y_X.log_prob(y)

            # The ratio of gradient is the same as the cross-entropy (for fixed entropy)
            ratio_grad = log_r_Y_X.mean()

            h_y = None
            if self.p_y is not None:
                # Compute the entropy y
                h_y = -self.p_y.log_prob(y).mean()
            elif self.h_y is not None:
                h_y = self.h_y

            if h_y is not None:
                # the ratio value is given by the difference between entropy and cross entropy
                ratio_value = (log_r_Y_X + h_y).mean()
            else:
                ratio_value = None

        return ratio_value, ratio_grad

    def compute_dual_ratio(self, x: torch.Tensor, y: torch.Tensor, y_: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], torch.Tensor]:
        # Computation of gradient and value of E_{p(x,y)}[f(x,y)]-log E_{r(x,y)}[e^{f(x,y)}]
        if self.ratio_estimator is None:
            ratio_value, ratio_grad = torch.zeros(1).to(x.device), torch.zeros(1).to(x.device)
        else:
            # Compute the ratio f(x,y) on samples from p(x,y). The expected shape is [N, M]
            f = self.ratio_estimator(x, y)

            # Negative samples from r(y|x)
            if y_ is None or self.predictive is None:
                y_ = self.sample_proposal(x, y)

            # Compute the ratio on the samples from the proposal [N, M']
            f_ = self.ratio_estimator(x, y_)

            ratio_value = self.compute_dual_ratio_value(x, y, f, f_)
            ratio_grad = self.compute_dual_ratio_grad(x, y, f, f_)

            if ratio_grad is None:
                ratio_grad = ratio_value

        return ratio_value, ratio_grad

    @staticmethod
    def _compute_dual_ratio_value(x, y, f, f_, baseline):
        if baseline is not None:
            b = baseline(f_, x, y)
            if b.ndim == 1:
                b = b.unsqueeze(1)
            assert b.ndim == f_.ndim
        else:
            b = torch.zeros_like(x)

        f = f - b

        if isinstance(baseline, BatchLogMeanExp):
            Z = 1
        else:
            Z = f_.exp().mean(1).unsqueeze(1)/b.exp()

        ratio_value = f - Z + 1
        return ratio_value.mean()
    def compute_dual_ratio_value(self, x, y, f, f_):
        return self._compute_dual_ratio_value(x, y, f, f_, self.baseline)

    def compute_dual_ratio_grad(self, x, y, f, f_):
        if self.grad_baseline is None:
            return None
        else:
            return self._compute_dual_ratio_value(x, y, f, f_, self.grad_baseline)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None,
            a: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Compute a lower bound for I(x,y).
        Args:
            x: a tensor with shape [N, D] in which x[i] is sampled from p(x)
            y: a tensor with shape [N, D] or [N, M, D] in which y[i,j] is sampled from p(y|x[i])
            y_: a tensor with shape [N, D] or [N, M', D] in which y[i,j] is sampled from r(y|x[i])
            a: a tensor with shape [N, D] representing the "attributes" corresponding to x
        Returns:
            mi_value, mi_grad: A tuple consisting of 1) the estimation for I(x,y) and 2) a quantity to differentiate to
                maximize mutual information. Note that 1) and 2) can have different values.
        """

        if y.ndim == x.ndim:
            # If one dimension is missing, we assume there is only one positive sample
            y = y.unsqueeze(1)

        assert y.ndim == x.ndim + 1

        if y_ is not None:
            if y_.ndim == x.ndim:
                # If one dimension is missing, we assume there is only one negative sample
                y_ = y_.unsqueeze(1)
            assert y_.ndim == x.ndim + 1

        # Compute the ratio using the primal bound
        primal_value, primal_grad = self.compute_primal_ratio(x, y, a)

        dual_value, dual_grad = self.compute_dual_ratio(x, y, y_)

        mi_grad = primal_grad + dual_grad
        if primal_value is not None:
            mi_value = primal_value + dual_value
        else:
            mi_value = None

        return mi_value, mi_grad
