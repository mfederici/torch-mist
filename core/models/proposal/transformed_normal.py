from typing import Dict, List, Any, Optional

import pyro.distributions.transforms as pyro_transforms_module
import core.distributions.transforms as transforms_module

from core.distributions.transforms import ConditionalTransformedDistributionModule
from core.distributions.parametrizations import ParametrizedNormal, ConstantParametrizedConditionalDistribution


class TransformedNormalProposal(ConditionalTransformedDistributionModule):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dims: Optional[List[int]] = None,
        transform_name: str = "conditional_linear",
        n_transforms: int = 1,
        transform_params: Dict[Any, Any] = None
    ):
        if transform_params is None:
            transform_params = {}

        base = ConstantParametrizedConditionalDistribution(input_dim=y_dim, parametrization=ParametrizedNormal())
        transforms = []
        if hasattr(pyro_transforms_module, transform_name):
            transform_factory = getattr(pyro_transforms_module, transform_name)
        elif hasattr(transforms_module, transform_name):
            transform_factory = getattr(transforms_module, transform_name)
        else:
            raise NotImplementedError(
                f"Transform {transform_name} is not implemented."
            )
        for transform in range(n_transforms):
            transform = transform_factory(input_dim=y_dim, context_dim=x_dim, hidden_dims=hidden_dims,
                                          **transform_params)
            transforms.append(transform)

        super(TransformedNormalProposal, self).__init__(base_dist=base, transforms=transforms)



