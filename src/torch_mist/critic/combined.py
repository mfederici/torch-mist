import torch

from torch_mist.critic import Critic
from torch_mist.estimators import MIEstimator


class CombinedCritic(Critic):
    def __init__(self, estimator: MIEstimator, base_critic: Critic):
        super().__init__()
        self.estimator = estimator
        self.base_critic = base_critic

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim == y.ndim - 1:
            x = x.unsqueeze(0)
        assert x.shape[1:-1] == y.shape[1:-1]
        f1 = self.base_critic(x, y)
        f2 = self.estimator.log_ratio(x, y)
        assert f1.shape == f2.shape
        return f1 + f2
