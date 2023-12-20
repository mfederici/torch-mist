from typing import Callable, Any

from torch import nn


class DummyModule(nn.Module):
    def __init__(self, f: Callable[[Any], Any]):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __repr__(self):
        return str(self.f)
