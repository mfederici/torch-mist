from typing import Callable, Optional

from pyro.distributions import Delta

from torch_mist.distributions.transforms import ConditionalDistributionModule


class ConditionalDelta(ConditionalDistributionModule):
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform

    def condition(self, context):
        if not (self.transform is None):
            context = self.transform(context)
        return Delta(context)
