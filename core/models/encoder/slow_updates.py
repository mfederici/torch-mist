from copy import deepcopy

import torch
from torch import nn


class SlowlyUpdatingModel(nn.Module):
    def __init__(self, model: nn.Module, tau: float):
        super(SlowlyUpdatingModel, self).__init__()
        self._current_model = [model] # Trick to not register the parameters of the model
        self.slow_model = deepcopy(model)
        for param in self.slow_model.parameters():
            param.requires_grad = False

        assert 0 <= tau <= 1
        self.tau = tau

    @property
    def current_model(self) -> nn.Module:
        return self._current_model[0]

    def _update_weights(self):
        """Update target network parameters."""
        for current_p, slow_p in zip(self.current_model.parameters(), self.slow_model.parameters()):
            slow_p.data = self.tau * slow_p.data + (1.0 - self.tau) * current_p.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.slow_model(x)

        if self.current_model.training:
            self._update_weights()

        return y
