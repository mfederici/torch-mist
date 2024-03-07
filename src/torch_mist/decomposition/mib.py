from typing import Dict, Any, Union, Optional

import numpy as np
import torch
from torch import nn

from torch_mist.decomposition.base import (
    DimensionalityReduction,
    StochasticDimensionalityReduction,
)
from torch_mist.distributions import (
    NormalModule,
    conditional_transformed_normal,
)
from torch_mist.estimators import instantiate_estimator
import torch_mist.models as tmm
from torch_mist.nn import dense_nn
from torch_mist.utils.data.utils import TensorDictLike
from torch_mist.utils.train import train_model

DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_BATCH_SIZE = 64


class MIB(StochasticDimensionalityReduction):
    def __init__(self, *args, beta: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def _add_default_mi_estimator_params(
        self, model_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not ("hidden_dims" in model_params):
            model_params["hidden_dims"] = [128, 64]

        if not ("estimator_name" in model_params):
            model_params["estimator_name"] = "infonce"
            if not ("k_dim" in model_params):
                model_params["k_dim"] = 64
        return model_params

    def _add_default_model_params(
        self, model_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if model_params is None:
            model_params = {}

        if not ("mi_estimator" in model_params):
            model_params["mi_estimator"] = {}

        model_params["mi_estimator"] = self._add_default_mi_estimator_params(
            model_params["mi_estimator"]
        )

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

    def _add_default_y_proj_params(
        self, proj_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if proj_params is None:
            proj_params = {}
        if not ("hidden_dims" in proj_params):
            proj_params["hidden_dims"] = [128]
        if not ("nonlinearity" in proj_params):
            proj_params["nonlinearity"] = nn.ReLU(True)

        return proj_params

    def _instantiate_y_proj(self, y_dim: int) -> nn.Module:
        self.proj_params["input_dim"] = self.n_dim
        self.proj_params["context_dim"] = y_dim
        return conditional_transformed_normal(**self.proj_params)

    def _train_model(self, data: TensorDictLike, **train_params):
        return train_model(
            self.model,
            data,
            eval_method="loss",
            train_logged_methods=["regularization", "loss"],
            eval_logged_methods=["regularization", "mutual_information"],
            **train_params
        )

    def _instantiate_model(self, x_dim: int, y_dim: int):
        self.model_params["mi_estimator"]["x_dim"] = self.n_dim
        self.model_params["mi_estimator"]["y_dim"] = self.n_dim
        self.model_params["mi_estimator"] = instantiate_estimator(
            **self.model_params["mi_estimator"]
        )

        return tmm.bottleneck.MIB(
            p_ZX_given_X=self.proj,
            p_ZY_given_Y=self.y_proj,
            beta=self.beta,
            **self.model_params
        )
