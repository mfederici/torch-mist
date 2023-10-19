from typing import List, Optional

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Categorical, Distribution

from torch_mist.distributions.utils import CategoricalModule
from torch_mist.estimators.discriminative.base import EmpiricalDistribution
from torch_mist.estimators.generative.base import (
    GenerativeMIEstimator,
)
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.caching import cached


class SameBucketConditionalDistribution(ConditionalDistribution):
    # Technically this is not a lower-bound but the estimation of marginal entropy is usually accurate
    lower_bound = True

    def __init__(self, Q: QuantizationFunction):
        self.Q = Q

    def condition(self, context: torch.Tensor):
        q = self.Q(context)
        q_0 = q.view(-1)[0]
        # Check all the elements are the same
        assert (
            q == q_0
        ).sum() == q.numel(), (
            "All elements of the quantized context must be the same"
        )
        return EmpiricalDistribution()


class PQ(GenerativeMIEstimator):
    def __init__(
        self,
        q_QX_given_Y: ConditionalDistribution,
        Q_x: QuantizationFunction,
        temperature: float = 1.0,
    ):
        super().__init__(q_Y_given_X=SameBucketConditionalDistribution(Q=Q_x))

        self.q_QX_given_Y = q_QX_given_Y
        self.q_QX = CategoricalModule(
            torch.zeros(Q_x.n_bins), temperature=temperature, learnable=True
        )
        self.Q_x = Q_x

    @cached
    def quantize_x(self, x: torch.Tensor) -> torch.Tensor:
        return self.Q_x(x)

    @cached
    def q_QX_given_y(self, y: torch.Tensor) -> Distribution:
        q_QX_given_y = self.q_QX_given_Y.condition(y)
        assert isinstance(q_QX_given_y, Categorical)
        return q_QX_given_y

    @cached
    def approx_log_p_qx_given_y(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # probabilities for the categorical distribution over q(y), shape [N, N_BINS]
        qx = self.quantize_x(x=x)

        q_QX_given_y = self.q_QX_given_y(y=y)

        # log-probabilities for the categorical distribution r(q(y)|x), shape [N, N_BINS]
        log_q_qx_given_y = q_QX_given_y.log_prob(qx)

        return log_q_qx_given_y

    @cached
    def approx_log_p_qx(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qx = self.quantize_x(x=x)
        log_q_qx = self.q_QX.log_prob(qx)

        return log_q_qx

    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.approx_log_p_qx_given_y(x=x, y=y) - self.approx_log_p_qx(
            x=x, y=y
        )

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q_qx_given_y = self.approx_log_p_qx_given_y(x=x, y=y)
        q_qx = self.approx_log_p_qx(x=x, y=y)
        return -(q_qx_given_y + q_qx).mean()

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += (
            f"  (q_QX_given_Y): "
            + self.q_QX_given_Y.__repr__().replace("\n", "\n  ")
            + "\n"
        )
        s += f"  (Q_x): " + self.Q_x.__repr__().replace("\n", "\n  ") + "\n"
        s += ")"
        return s


def pq(
    Q_x: QuantizationFunction,
    y_dim: Optional[int] = None,
    hidden_dims: Optional[List[int]] = None,
    q_QX_given_Y: Optional[ConditionalDistribution] = None,
    temperature: float = 0.1,
) -> PQ:
    from torch_mist.distributions.utils import conditional_categorical

    if q_QX_given_Y is None:
        if y_dim is None or hidden_dims is None:
            raise ValueError(
                "Either q_qY_given_X or y_dim and hidden_dims must be specified."
            )
        q_QX_given_Y = conditional_categorical(
            n_classes=Q_x.n_bins,
            context_dim=y_dim,
            hidden_dims=hidden_dims,
            temperature=temperature,
        )

    return PQ(
        q_QX_given_Y=q_QX_given_Y,
        Q_x=Q_x,
        temperature=temperature,
    )
