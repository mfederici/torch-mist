from typing import Union, List, Optional, Dict
from abc import abstractmethod

import torch


class JointDistribution:
    labels = []

    def entropy(self, labels: Union[str, List[str]]) -> Optional[float]:
        return None

    @abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()