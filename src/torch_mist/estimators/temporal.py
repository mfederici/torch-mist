from typing import Union, List
import numpy as np
import torch
from copy import deepcopy

from torch_mist.estimators.multi import MultiMIEstimator
from torch_mist.estimators.base import MIEstimator


class TemporalMIEstimator(MultiMIEstimator):
    def __init__(
        self,
        base_estimator: MIEstimator,
        lagtimes: Union[List[int], np.ndarray, torch.Tensor],
    ):
        super().__init__(
            estimators={
                ("t_0", f"t_{int(lagtime)}"): deepcopy(base_estimator)
                for lagtime in lagtimes
            }
        )
