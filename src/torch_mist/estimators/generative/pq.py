from typing import List, Optional

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Categorical, Distribution

from torch_mist.distributions.utils import CategoricalModule
from torch_mist.estimators.generative.doe import DoE
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.caching import cached


class PQ(DoE):
    def __init__(
            self,
            q_qY_given_X: ConditionalDistribution,
            quantization: QuantizationFunction,
            temperature: float = 0.1,
    ):
        super().__init__(
            q_Y_given_X=q_qY_given_X, 
            q_Y=CategoricalModule(torch.zeros(quantization.n_bins), temperature=temperature, learnable=True)
        )
        self.quantization=quantization

    @cached
    def quantize_y(self, y: torch.Tensor) -> torch.Tensor:
        return self.quantization(y)

    @cached
    def q_Y_given_x(self, x: torch.Tensor) -> Distribution:
        q_qY_given_x = self.q_Y_given_X.condition(x)
        assert isinstance(q_qY_given_x, Categorical)
        return q_qY_given_x

    @cached
    def approx_log_p_y_given_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # probabilities for the categorical distribution over q(y), shape [N, N_BINS]
        y = self.quantize_y(y=y)

        q_qy_given_x = self.q_Y_given_x(x=x)

        assert isinstance(q_qy_given_x, Categorical)
        # log-probabilities for the categorical distribution r(q(y)|x), shape [N, N_BINS]
        log_q_qy_x = q_qy_given_x.logits - torch.logsumexp(q_qy_given_x.logits, dim=-1, keepdim=True)

        assert log_q_qy_x.ndim == y.ndim, f'log_r_qy_x.shape={log_q_qy_x.shape}'

        # E_q(Q(y)|y)[log q(Q(y)|x)]
        log_q_qy_given_x = torch.sum(y * log_q_qy_x, -1)
        return log_q_qy_given_x

    @cached
    def approx_log_p_y(self, y: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # probabilities for the categorical distribution over q(y), shape [N, BINS]
        y = self.quantize_y(y=y)

        logits = self.q_Y.logits/self.q_Y.temperature

        # log-probabilities for the marginal categorical distribution q(q(y)), shape [1, N_BINS]
        log_q_qy = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        while log_q_qy.ndim < y.ndim:
            log_q_qy = log_q_qy.unsqueeze(0)

        assert log_q_qy.ndim == y.ndim, f'log_q_qy.shape={log_q_qy.shape}'

        return torch.sum(y * log_q_qy, -1)


def pq(
        x_dim: int,
        hidden_dims: List[int],
        quantization: QuantizationFunction,
) -> PQ:
    from torch_mist.distributions.utils import conditional_categorical

    q_y_x = conditional_categorical(
        n_classes=quantization.n_bins,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
    )

    return PQ(
        q_qY_given_X=q_y_x,
        quantization=quantization,
    )
