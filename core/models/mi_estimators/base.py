from typing import Tuple, Optional

import torch
import torch.nn as nn

# The implementations in this work are generally based on "On Variational Lower bounds of mutual information"
# https://arxiv.org/pdf/1905.06922.pdf
from core.logging import LoggingModule


# Mutual Information Estimation Based on the dual representation of a KL divergence
class MutualInformationEstimator(nn.Module):
    def compute_ratio(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # Computation of gradient and value of E_{p(x,y)}[log r(y|x)/p(y)]
        # By default we consider r(y|x) = p(y), therefore E_{p(x,y)}[log r(y|x)/p(y)] = 0
        return torch.zeros(1).to(x.device), torch.zeros(1).to(x.device)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            y_: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Compute a lower bound for I(x,y).
        Args:
            x: a tensor with shape [N, D] in which x[i] is sampled from p(x)
            y: a tensor with shape [N, D] or [N, M, D] in which y[i,j] is sampled from p(y|x[i])
            y_: a tensor with shape [N, D] or [N, M', D] in which y[i,j] is sampled from r(y|x[i])
        Returns:
            mi_value, mi_grad: A tuple consisting of 1) the estimation for I(x,y) and 2) a quantity to differentiate to
                maximize mutual information. Note that 1) and 2) can have different values.
        """

        if y.ndim == x.ndim:
            # If one dimension is missing, we assume there is only one positive sample
            y = y.unsqueeze(1)

        assert y.ndim == x.ndim + 1


        # We compute a lower-bound of I(x;y)
        mi_value, mi_grad = self.compute_ratio(x, y, y_)

        assert mi_grad.shape[0]==x.shape[0]
        assert mi_grad.shape[1]==y.shape[1]

        return mi_value.mean(), mi_grad.mean()
