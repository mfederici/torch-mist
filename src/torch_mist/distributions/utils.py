from typing import Dict, List, Any, Optional
import inspect

from pyro.nn import DenseNN

from torch.distributions import Distribution, Independent


import torch
from torch import nn
from torch.distributions import Normal, Categorical


from torch_mist.distributions.transforms import ConditionalTransformedDistributionModule, DistributionModule, \
    TransformedDistributionModule, ConditionalDistributionModule
from torch_mist.distributions.parametrizations.map import LocScaleMap, LogitsMap


def fetch_transform(transform_name: str):
    import pyro.distributions.transforms as pyro_transforms_module
    import torch_mist.distributions.transforms as transforms_module

    if hasattr(pyro_transforms_module, transform_name):
        transform_factory = getattr(pyro_transforms_module, transform_name)
    elif hasattr(transforms_module, transform_name):
        transform_factory = getattr(transforms_module, transform_name)
    else:
        raise NotImplementedError(
            f"Transform {transform_name} is not implemented."
        )
    return transform_factory


class ConditionalCategoricalModule(ConditionalDistributionModule):
    def __init__(self, net: nn.Module):
        super(ConditionalCategoricalModule, self).__init__()
        self.net = net
        self.parametrization = LogitsMap()

    def condition(self, x):
        return Categorical(**self.parametrization(self.net(x)))


class NormalModule(Distribution, nn.Module):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, learnable: bool = False):
        assert loc.ndim == 1
        nn.Module.__init__(self)
        Distribution.__init__(
            self,
            event_shape=torch.Size([loc.shape[0]]),
            validate_args=False
        )
        if learnable:
            self.loc = nn.Parameter(loc)
            self.log_scale = nn.Parameter(scale.log())
        else:
            self.register_buffer('loc', loc)
            self.register_buffer('log_scale', scale.log())
        self.parametrization = LocScaleMap()

    def rsample(self, sample_shape=torch.Size()):
        params = self.parametrization([self.loc, self.log_scale])
        return Normal(**params).rsample(sample_shape)

    def log_prob(self, value):
        params = self.parametrization([self.loc, self.log_scale])
        return Independent(Normal(**params), 1).log_prob(value)

    def __repr__(self):
        return "Normal()"


class CategoricalModule(Distribution, nn.Module):
    def __init__(self, logits: torch.tensor, temperature: float = 1.0, learnable: bool = False):
        nn.Module.__init__(self)
        Distribution.__init__(
            self,
            event_shape=torch.Size([logits.shape[0]]),
            validate_args=False
        )
        if learnable:
            self.logits = nn.Parameter(logits)
        else:
            self.register_buffer('logits', logits)
        self.parametrization = LogitsMap()
        self.temperature = temperature

    def rsample(self, sample_shape=torch.Size()):
        params = self.parametrization([self.logits/self.temperature])
        return Categorical(**params).rsample(sample_shape)

    def log_prob(self, value):
        params = self.parametrization([self.logits/self.temperature])
        return Categorical(**params).log_prob(value)

    def __repr__(self):
        return "Categorical()"


class ConditionalStandardNormalModule(ConditionalDistributionModule):
    def __init__(self, n_dim: int):
        super().__init__()
        self.register_buffer('loc', torch.zeros(n_dim))
        self.register_buffer('log_scale', torch.zeros(n_dim))
        self.parametrization = LocScaleMap()

    def condition(self, context):
        extra_dims = context.ndim - self.loc.ndim
        loc, log_scale = self.loc, self.log_scale
        for _ in range(extra_dims):
            loc = loc.unsqueeze(0)
            log_scale = log_scale.unsqueeze(0)

        loc = loc.expand(*context.shape[:-1], -1)
        log_scale = log_scale.expand(*context.shape[:-1], -1)

        return Independent(Normal(**self.parametrization([loc, log_scale])), 1)


def conditional_transformed_normal(
    input_dim: int,
    context_dim: int,
    hidden_dims: Optional[List[int]] = None,
    transform_name: str = "conditional_linear",
    n_transforms: int = 1,
    transform_params: Dict[Any, Any] = None
):
    if transform_params is None:
        transform_params = {}

    assert n_transforms > 0, "n_transforms must be greater than 0"

    base = ConditionalStandardNormalModule(input_dim)
    transforms = []

    transform_factory = fetch_transform(transform_name)

    for transform in range(n_transforms):
        if "hidden_dims" in inspect.signature(transform_factory).parameters:
            transform_params.update({"hidden_dims": hidden_dims})

        transform = transform_factory(
            input_dim=input_dim,
            context_dim=context_dim,
            **transform_params
        )

        transforms.append(transform)

    return ConditionalTransformedDistributionModule(base_dist=base, transforms=transforms)


def transformed_normal(
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        transform_name: str = "linear",
        n_transforms: int = 1,
        transform_params: Dict[Any, Any] = None
) -> TransformedDistributionModule:
    if transform_params is None:
        transform_params = {}

    assert n_transforms > 0, "n_transforms must be greater than 0"

    base_dist = NormalModule(torch.zeros(input_dim), torch.ones(input_dim))
    transforms = []

    transform_factory = fetch_transform(transform_name)

    for transform in range(n_transforms):
        if "hidden_dims" in inspect.signature(transform_factory).parameters:
            transform_params.update({"hidden_dims": hidden_dims})

        transform = transform_factory(
            input_dim=input_dim,
            **transform_params
        )

        transforms.append(transform)

    return TransformedDistributionModule(
        base_dist=base_dist,
        transforms=transforms,
    )


def conditional_categorical(
    n_classes: int,
    context_dim: int,
    hidden_dims: List[int],
):
    net = DenseNN(input_dim=context_dim, hidden_dims=hidden_dims, param_dims=[n_classes])
    return ConditionalCategoricalModule(net)
