from typing import Optional

import torch
from torch import nn

from torch_mist.estimators.generative.base import (
    GenerativeMIEstimator,
)
from torch_mist.estimators.generative.pq import (
    SameBucketConditionalDistribution,
)
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.caching import (
    cached,
    reset_cache_after_call,
    reset_cache_before_call,
)


class DiscreteMIEstimator(GenerativeMIEstimator):
    # Technically this is not a lower-bound but the estimation of marginal entropy is usually accurate
    lower_bound = True

    def __init__(
        self,
        Q_x: Optional[QuantizationFunction] = None,
        Q_y: Optional[QuantizationFunction] = None,
        temperature: float = 1.0,
    ):
        self.x_bins = Q_x.n_bins
        self.y_bins = Q_y.n_bins
        super().__init__(q_Y_given_X=SameBucketConditionalDistribution(Q=Q_x))

        self.Q_x = Q_x
        self.Q_y = Q_y

        # Start from uniform distribution
        self._logits = nn.Parameter(
            torch.zeros(Q_x.n_bins, Q_y.n_bins),
        )

        self.temperature = temperature

    @cached
    def quantize_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.Q_x is not None:
            x = self.Q_x(x)
        else:
            assert isinstance(x, torch.LongTensor)
        return x

    @cached
    def quantize_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.Q_y is not None:
            y = self.Q_y(y)
        else:
            assert isinstance(y, torch.LongTensor)
        return y

    @property
    def logits(self) -> torch.Tensor:
        logits = self._logits / self.temperature
        return logits - logits.logsumexp(dim=[0, 1], keepdim=True)

    @cached
    def approx_log_p_qxqy(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        qx = self.quantize_x(x=x)
        qy = self.quantize_y(y=y)

        # Expand to shape [..., x_bins, y_bins] with one hot encoding
        qxqy = torch.einsum(
            "...x, ...y -> ...xy",
            torch.nn.functional.one_hot(qx, self.x_bins).float(),
            torch.nn.functional.one_hot(qy, self.y_bins).float(),
        ).float()

        # compute the log probability of each sample under the model
        log_q_qxqy = torch.einsum("...xy, xy -> ...", qxqy, self.logits)

        assert log_q_qxqy.ndim == y.ndim - 1
        return log_q_qxqy

    def approx_log_p_qx(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qx = self.quantize_x(x=x)

        log_q_qx = torch.einsum(
            "...x, x -> ...",
            torch.nn.functional.one_hot(qx, self.x_bins).float(),
            torch.logsumexp(self.logits, -1),
        )
        assert (
            log_q_qx.shape == x.shape[:-1]
        ), f"{log_q_qx.shape} != {x.shape[:-1]}"

        return log_q_qx

    def approx_log_p_qy(
        self, y: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        qy = self.quantize_y(y=y)

        log_q_qy = torch.einsum(
            "...y, y -> ...",
            torch.nn.functional.one_hot(qy, self.y_bins).float(),
            torch.logsumexp(self.logits, -2),
        )
        assert (
            log_q_qy.shape == y.shape[:-1]
        ), f"{log_q_qy.shape} != {y.shape[:-1]}"

        return log_q_qy

    def log_p_qx_given_qy(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.approx_log_p_qxqy(x=x, y=y) - self.approx_log_p_qy(y=y)

    def approx_log_p_qy_given_qx(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.approx_log_p_qxqy(x=x, y=y) - self.approx_log_p_qx(x=x)

    @reset_cache_before_call
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = -self.approx_log_p_qxqy(x=x, y=y)
        return loss.mean()

    @reset_cache_after_call
    def log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_qx_given_qy = self.log_p_qx_given_qy(x=x, y=y)
        log_qx = self.approx_log_p_qx(x=x)
        return log_qx_given_qy - log_qx

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += f"  (Q_x): " + self.Q_x.__repr__().replace("\n", "\n  ") + "\n"
        s += f"  (Q_y): " + self.Q_y.__repr__().replace("\n", "\n  ") + "\n"
        s += ")"
        return s


def discrete(
    Q_x: Optional[QuantizationFunction] = None,
    Q_y: Optional[QuantizationFunction] = None,
    temperature: float = 0.1,
) -> DiscreteMIEstimator:
    return DiscreteMIEstimator(
        Q_x=Q_x,
        Q_y=Q_y,
        temperature=temperature,
    )
