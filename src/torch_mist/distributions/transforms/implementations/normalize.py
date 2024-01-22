import torch

from torch_mist.distributions.transforms.implementations.linear import Linear


class EMANormalize(Linear):
    def __init__(
        self,
        input_dim: int,
        epsilon: float = 1e-6,
        gamma: float = 0.99,
        normalize_inverse=True,
    ):
        super().__init__(input_dim=input_dim, epsilon=epsilon)

        assert 0 <= gamma <= 1
        self.gamma = gamma
        self.normalize_inverse = normalize_inverse

    def _update_params(self, t: torch.Tensor):
        self.loc = (
            self.gamma * self.loc + (1 - self.gamma) * t.mean(0).detach()
        )
        self.log_scale = (
            self.gamma * self.log_scale.exp()
            + (1 - self.gamma)
            * torch.clamp(t.std(0).detach(), min=self.epsilon)
        ).log()

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.training and self.normalize_inverse:
            self._update_params(y)
        return super()._inverse(y=y)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.normalize_inverse:
            self._update_params(x)
        return super()._call(x=x)
