from typing import List, Dict, Union

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch import nn

from torch_mist.distributions.transforms import ConditionalDistributionModule


class JointDistribution(nn.Module):
    def __init__(self, joint_dist: Distribution, dims: List[int], names: List[str]):
        super().__init__()
        self.joint_dist = joint_dist
        self.dims = dims
        self.names = names

    def log_prob(self, *args, **kwargs) -> torch.Tensor:
        # Add the args to kwargs
        unused_names = [name for name in self.names if not (name in kwargs)]
        new_kwargs = {unused_names[i]: arg for i, arg in enumerate(args)}

        kwargs = {**kwargs, **new_kwargs}

        # Expand args to the same shape if needed
        assert len(kwargs) == len(self.dims), f"passed: {kwargs.keys()} != required: {self.names}"

        args = list(kwargs.values())
        n_dims = args[0].ndim
        # Find the maximum shape
        max_shape = [max([arg.shape[i] for arg in args]) for i in range(n_dims-1)]
        # Expand all args to the maximum shape
        kwargs = {name: value.expand(max_shape + [-1]) for name, value in kwargs.items()}

        # Concatenate all the arguments in order
        args = torch.cat([kwargs[name] for name in self.names], dim=-1)

        # Compute the log prob
        log_prob = self.joint_dist.log_prob(args)
        assert log_prob.shape == args.shape[:-1]
        return log_prob

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Dict[str, torch.Tensor]:
        sample = self.joint_dist.sample(sample_shape)
        sample = torch.split(sample, self.dims, dim=-1)
        return {name: value.squeeze(-1) for name, value in zip(self.names, sample)}

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Dict[str, torch.Tensor]:
        rsample = self.joint_dist.rsample(sample_shape)
        rsample = torch.split(rsample, self.dims, dim=-1)
        return {name: value.squeeze(-1) for name, value in zip(self.names, rsample)}


class ConditionedRatioDistribution(Distribution):
    def __init__(
            self,
            joint: JointDistribution,
            log_marginal: torch.Tensor,
            conditioning: Dict[str, torch.Tensor]
    ):
        super().__init__(validate_args=False)
        self.joint = joint
        self.log_marginal = log_marginal
        self.conditioning = conditioning

    def log_prob(self, *args, **kwargs) -> torch.Tensor:
        kwargs = {**self.conditioning, **kwargs}
        log_joint = self.joint.log_prob(*args, **kwargs)
        log_conditional = log_joint - self.log_marginal
        return log_conditional


class ConditionalRatioDistribution(ConditionalDistributionModule):
    def __init__(self, joint: JointDistribution, marginal: Union[JointDistribution, Distribution]):
        super().__init__()
        self.joint = joint
        self.marginal = marginal

    def condition(self, *args, **kwargs) -> ConditionedRatioDistribution:
        # Add the args to kwargs
        unused_names = [name for name in self.joint.names if not (name in kwargs)]
        new_kwargs = {unused_names[i]: arg for i, arg in enumerate(args)}

        kwargs = {**kwargs, **new_kwargs}

        if isinstance(self.marginal, Distribution):
            assert len(kwargs) == 1
            context = list(kwargs.values())[0]
            log_marginal = self.marginal.log_prob(context)
        else:
            log_marginal = self.joint.log_prob(**kwargs)

        return ConditionedRatioDistribution(joint=self.joint, log_marginal=log_marginal, conditioning=kwargs)
