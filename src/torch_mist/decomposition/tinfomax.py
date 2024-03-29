from typing import Dict, Any, Union, Optional

import numpy as np
import torch
from torch import nn

from torch_mist.decomposition import MID
from torch_mist.estimators import (
    instantiate_estimator,
    TransformedMIEstimator,
)
from torch_mist.nn import Model

DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_BATCH_SIZE = 64


class TInfoMax(MID):
    def __init__(
        self,
        n_dim: int,
        lagtime: int,
        normalize_inputs: bool = True,
        whiten: bool = False,
        proj: Optional[nn.Module] = None,
        model: Optional[Model] = None,
        proj_params: Optional[Dict[str, Any]] = None,
        y_proj_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(lagtime, int) or lagtime <= 0:
            raise ValueError(f"Invalid lagtime {lagtime}.")

        super().__init__(
            n_dim=n_dim,
            normalize_inputs=normalize_inputs,
            whiten=whiten,
            proj=proj,
            proj_params=proj_params,
            model=model,
            y_proj_params=y_proj_params,
            model_params=model_params,
        )

        self.lagtime = lagtime

    def _instantiate_model(self, *args, **kwargs):
        self.model_params["x_dim"] = self.n_dim
        self.model_params["y_dim"] = self.n_dim
        mi_estimator = instantiate_estimator(**self.model_params)

        transforms = {"x": self.proj, "y": self.proj}

        return TransformedMIEstimator(
            transforms=transforms, base_estimator=mi_estimator
        )

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **train_params,
    ):
        XT1 = X[self.lagtime :]
        XT2 = X[: -self.lagtime]

        return super().fit(XT1, XT2, **train_params)
