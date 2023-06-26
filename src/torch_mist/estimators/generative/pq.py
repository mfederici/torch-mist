from typing import List

import torch
from pyro.distributions import ConditionalDistribution
from torch import nn
from torch.distributions import Categorical

from torch_mist.estimators.generative.base import GenerativeMutualInformationEstimator
from torch_mist.estimators.base import Estimation
from torch_mist.quantization import QuantizationFunction


class PQ(GenerativeMutualInformationEstimator):
    def __init__(
            self,
            conditional_qy_x: ConditionalDistribution,
            quantization: QuantizationFunction,
            temperature: float = 0.1,
    ):
        super().__init__()
        self.quantization = quantization
        self.qy_logits = nn.Parameter(torch.zeros(quantization.n_bins))
        self.conditional_qy_x = conditional_qy_x
        self.temperature = temperature

    def log_ratio(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> Estimation:
        # probabilities for the categorical distribution over q(y), shape [N, M, BINS]
        p_qy_y = self.quantization(y).detach()

        r_qy_x = self.conditional_qy_x.condition(x)
        assert isinstance(r_qy_x, Categorical)
        # log-probabilities for the categorical distribution r(q(y)|x), shape [N, 1, BINS]
        log_r_qy_x = r_qy_x.logits - torch.logsumexp(r_qy_x.logits, dim=-1, keepdim=True)


        # log-probabilities for the marginal categorical distribution r(q(y)), shape [1, 1, BINS]
        log_r_qy = self.qy_logits/self.temperature - torch.logsumexp(self.qy_logits/self.temperature, dim=-1, keepdim=True)
        log_r_qy = log_r_qy.unsqueeze(0).unsqueeze(0)

        # log-ratio, shape [N, M]
        log_ratio = torch.sum(p_qy_y * (log_r_qy_x - log_r_qy), -1)

        loss = -torch.sum(p_qy_y * log_r_qy_x, -1).mean()
        loss += -torch.sum(p_qy_y * log_r_qy, -1).mean()

        return Estimation(value=log_ratio, loss=loss)


def pq(
        x_dim: int,
        quantization: QuantizationFunction,
        hidden_dims: List[int],
) -> PQ:
    from torch_mist.distributions.utils import conditional_categorical

    q_y_x = conditional_categorical(
        n_classes=quantization.n_bins,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
    )

    return PQ(
        conditional_qy_x=q_y_x,
        quantization=quantization,
    )
