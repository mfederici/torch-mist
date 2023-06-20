import math
from typing import Optional, Dict

import torch
from torch import nn
from torch.distributions import Distribution, Categorical
from pyro.distributions import ConditionalDistribution

from src.torch_mist.distributions.joint.base import JointDistribution
from src.torch_mist.models.mi_estimator.base import MutualInformationEstimator, Estimation
from src.torch_mist.quantization import QuantizationFunction


class GenerativeMutualInformationEstimator(MutualInformationEstimator):
    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

    def log_prob_y_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplemented()

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        raise NotImplemented()

    def log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Estimation:
        # Compute the ratio using the primal bound
        estimates = {}

        x = x.unsqueeze(1)
        assert x.ndim == y.ndim

        log_p_y_x = self.log_prob_y_x(x, y)  # [N, M]
        log_p_y = self.log_prob_y(x, y)  # [N, M]

        assert log_p_y_x.ndim == log_p_y.ndim == 2, f'log_p_y_x.ndim={log_p_y_x.ndim}, log_p_y.ndim={log_p_y.ndim}'

        value = log_p_y_x - log_p_y
        loss = self.compute_loss(x=x, y=y, log_p_y_x=log_p_y_x, log_p_y=log_p_y)

        return Estimation(value=value, loss=loss)


class VariationalProposalMutualInformationEstimator(GenerativeMutualInformationEstimator):
    def __init__(self, conditional_y_x: ConditionalDistribution):
        super().__init__()
        self.conditional_y_x = conditional_y_x

        self._cached_p_y_X = None
        self._cached_x = None
        self._cached_y = None

    def log_prob_y_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute E[log r(y|x)]
        p_y_X = self.conditional_y_x.condition(x)
        log_p_Y_X = p_y_X.log_prob(y)

        assert log_p_Y_X.shape == y.shape[:-1]

        # Cache the conditional p(y|X=x) and the inputs x, y
        self._cached_p_y_X = p_y_X
        self._cached_x = x
        self._cached_y = y

        return log_p_Y_X

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return -log_p_y_x.mean()

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(conditional_y_x)=' + str(self.conditional_y_x).replace('\n', '  \n') + '\n'
        s += ')' + '\n'

        return s


class BA(VariationalProposalMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            marginal_y: Optional[Distribution] = None,
            H_y: Optional[torch.Tensor] = None,
    ):
        super().__init__(conditional_y_x)
        self.conditional_y_x = conditional_y_x
        assert (marginal_y is None) ^ (
                    H_y is None), 'Either the marginal distribution or the marginal entropy must be provided'

        self.marginal_y = marginal_y
        self.H_y = H_y

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.H_y is not None:
            return -torch.FloatTensor(self.H_y).unsqueeze(0).unsqueeze(1).to(y.device)
        else:
            log_q_y = self.marginal_y.log_prob(y)
            assert log_q_y.shape == y.shape[:-1]
            return log_q_y

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Optimize using maximum likelihood
        return -(log_p_y + log_p_y_x).mean()


class DoE(BA):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            marginal_y: Distribution,
    ):
        super().__init__(
            conditional_y_x=conditional_y_x,
            marginal_y=marginal_y,
        )

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(conditional_y_x)=' + str(self.conditional_y_x).replace('\n', '  \n') + '\n'
        s += '  ' + '(marginal_y)=' + str(self.marginal_y).replace('\n', '  \n') + '\n'
        s += ')' + '\n'

        return s


class GM(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            joint_xy: JointDistribution,
            marginal_y: Optional[Distribution] = None,
            H_y: Optional[torch.Tensor] = None,
            marginal_x: Optional[Distribution] = None,
            H_x: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.joint_xy = joint_xy
        assert (marginal_y is None) ^ (
                    H_y is None), 'Either the marginal distribution or the marginal entropy must be provided'
        assert (marginal_x is None) ^ (
                    H_x is None), 'Either the marginal distribution or the marginal entropy must be provided'
        self.marginal_y = marginal_y
        self.H_y = H_y
        self.marginal_x = marginal_x
        self.H_x = H_x

        self._cached_x = None
        self._cached_y = None
        self._cached_H_xy = None
        self._cached_H_x = None

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.H_y is not None:
            return -torch.FloatTensor(self.H_y).unsqueeze(0).unsqueeze(1).to(y.device)
        else:
            return self.marginal_y.log_prob(y).mean()

    def log_prob_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.H_x is not None:
            return -torch.FloatTensor(self.H_x).unsqueeze(0).unsqueeze(1).to(x.device)
        else:
            log_p_x = self.marginal_x.log_prob(x)
            if x.ndim == 2:
                log_p_x = log_p_x.unsqueeze(1)
            return log_p_x

    def log_prob_y_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute E[-log r(y|x)]
        log_r_XY = self.joint_xy.log_prob({'x': x, 'y': y})
        log_r_X = self.log_prob_x(x, y)
        log_r_Y_X = log_r_XY - log_r_X

        # Cache the entropy and the inputs x, y
        self._cached_x = x
        self._cached_y = y
        self._cached_H_xy = -log_r_XY.mean()
        self._cached_H_x = -log_r_X.mean()

        return log_r_Y_X

    def compute_loss(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            log_p_y: torch.Tensor,
            log_p_y_x: torch.Tensor,
    ):
        assert torch.equal(x, self._cached_x), 'The input x is not the same as the cached input x'
        assert torch.equal(y, self._cached_y), 'The input y is not the same as the cached input y'

        H_xy = self._cached_H_xy
        H_x = self._cached_H_x

        # Optimize using maximum likelihood
        return log_p_y.mean() + H_x + H_xy

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(joint_yx)=' + str(self.joint_xy).replace('\n', '  \n') + '\n'
        s += '  ' + '(marginal_x)=' + str(self.marginal_x).replace('\n', '  \n') + '\n'
        s += '  ' + '(marginal_y)=' + str(self.marginal_y).replace('\n', '  \n') + '\n'
        s += ')' + '\n'
        return s


class PQ(VariationalProposalMutualInformationEstimator):
    def __init__(
            self,
            conditional_qx_y: ConditionalDistribution,
            q: QuantizationFunction,
    ):
        super().__init__(conditional_qx_y)
        self.q = q
        self.y_logits = nn.Parameter(torch.zeros(q.num_bins))

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return Categorical(logits=self.y_logits).log_prob(y)

    def expected_log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Swap the order of x and y since we are quantizing and predicting the second argument
        return super().expected_log_ratio(y, self.q(x))

    def __repr__(self):
        s = self.__class__.__name__ + '(\n'
        s += '  ' + '(conditional_qx_y)=' + str(self.conditional_x_y).replace('\n', '  \n') + '\n'
        s += '  ' + '(q)=' + str(self.q).replace('\n', '  \n') + '\n'
        s += ')'
        return s


class CLUB(VariationalProposalMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
            sample: str = 'all'  # Use all the off-diagonal samples as negative samples
    ):
        super().__init__(conditional_y_x)
        assert sample in ['all', 'one', 'l1o'], 'The sample must be one of \n' \
                                                '  "all": use all samples in the batch as negatives\n' \
                                                '  "one": use one sample in the batch as negative\n' \
                                                '  "l1o": use all off-diagonal samples in the batch as negatives'
        self.sample = sample

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.equal(x, self._cached_x), 'The input x is not the same as the cached input x'
        assert torch.equal(y, self._cached_y), 'The input y is not the same as the cached input y'

        if self.sample == 'one':
            y_ = torch.roll(y, 1, dims=0)
            log_p_y = self._cached_p_y_X.log_prob(y_)
            log_p_y = log_p_y.mean(dim=0).unsqueeze(0)
        else:
            N = y.shape[0]
            assert y.shape[1] == 1, 'The CLUB estimator can only be used with a single y'
            y_ = y.squeeze(1).unsqueeze(0)
            log_p_y = self._cached_p_y_X.log_prob(y_)  # [N, N]

            if self.sample == 'l1o':
                # Remove the diagonal
                log_p_y = log_p_y * (1 - torch.eye(N).to(y.device))
                N = N - 1

            log_p_y = torch.sum(log_p_y, dim=0).unsqueeze(0)/N
        return log_p_y

class L1Out(VariationalProposalMutualInformationEstimator):
    def __init__(
            self,
            conditional_y_x: ConditionalDistribution,
    ):
        super().__init__(conditional_y_x)

    def log_prob_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.equal(x, self._cached_x), 'The input x is not the same as the cached input x'
        assert torch.equal(y, self._cached_y), 'The input y is not the same as the cached input y'


        N = y.shape[0]
        assert y.shape[1] == 1, 'The L1Out estimator can only be used with a single y'
        y_ = y.squeeze(1).unsqueeze(0)
        log_p_y = self._cached_p_y_X.log_prob(y_)  # [N, N]

        # Remove the diagonal
        log_p_y = log_p_y * (1 - torch.eye(N).to(y.device))
        log_p_y = log_p_y + torch.nan_to_num(torch.eye(N).to(y.device)*(-float('inf')), 0, float('inf'), -float('inf'))
        log_p_y = torch.logsumexp(log_p_y, dim=0).unsqueeze(0) - math.log(N-1)
        return log_p_y


