from typing import Dict, Any, Union, Optional

import numpy as np
import torch
from torch import nn

from torch_mist.decomposition.base import DimensionalityReduction
from torch_mist.estimators import (
    instantiate_estimator,
    TransformedMIEstimator,
)
from torch_mist.nn import dense_nn, Identity
from torch_mist.utils import train_mi_estimator
from torch_mist.utils.data.utils import TensorDictLike

DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_BATCH_SIZE = 64


class MID(DimensionalityReduction):
    def _add_default_model_params(
        self, model_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if model_params is None:
            model_params = {}
        if not ("hidden_dims" in model_params):
            model_params["hidden_dims"] = [128, 64]

        if not ("estimator_name" in model_params):
            model_params["estimator_name"] = "smile"
            if not ("neg_samples" in model_params):
                model_params["neg_samples"] = 8

        return model_params

    def _add_default_proj_params(
        self, proj_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if proj_params is None:
            proj_params = {}
        if not ("hidden_dims" in proj_params):
            proj_params["hidden_dims"] = [128]
        if not ("nonlinearity" in proj_params):
            proj_params["nonlinearity"] = nn.ReLU(True)

        return proj_params

    def _instantiate_proj(self, x_dim: int) -> nn.Module:
        self.proj_params["input_dim"] = x_dim
        self.proj_params["output_dim"] = self.n_dim

        return dense_nn(**self.proj_params)

    def _instantiate_y_proj(self, y_dim: int) -> Optional[nn.Module]:
        if not (self.y_proj_params is None):
            self.y_proj_params["input_dim"] = y_dim
            return dense_nn(**self.y_proj_params)
        else:
            return None

    def _train_model(self, data: TensorDictLike, **train_params):
        return train_mi_estimator(self.model, data, **train_params)

    def _instantiate_model(self, x_dim: int, y_dim: int):
        self.model_params["x_dim"] = self.n_dim
        self.model_params["y_dim"] = y_dim
        mi_estimator = instantiate_estimator(**self.model_params)

        transforms = {"x": self.proj}
        if not (self.y_proj is None):
            transforms["y"] = self.y_proj

        return TransformedMIEstimator(
            transforms=transforms, base_estimator=mi_estimator
        )
