from typing import Union, Dict

import torch

from torch_mist.distributions import JointDistribution


def prepare_samples(
    distribution: JointDistribution, n_samples: int
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    true_mi = distribution.mutual_information()
    print(f"True MI: {true_mi} nats")
    return distribution.sample(torch.Size([n_samples]))


def compute_mi(distribution: JointDistribution, **kwargs) -> float:
    return distribution.mutual_information(**kwargs).item()
