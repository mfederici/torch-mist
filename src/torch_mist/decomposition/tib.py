from typing import Dict, Any, Union, Optional, List

import numpy as np
import torch
from pyro.distributions import ConditionalDistribution
from torch import nn

from torch_mist.decomposition import CEB
from torch_mist.decomposition.base import (
    DimensionalityReduction,
    StochasticDimensionalityReduction,
    CenterAndScale,
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


class TIB(CEB):
    def __init__(
        self,
        n_dim: int,
        lagtime: int,
        transition_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            n_dim=n_dim,
            conditional_dist_params=transition_params,
        )

        if not isinstance(lagtime, int) or lagtime <= 0:
            raise ValueError(f"Invalid lagtime {lagtime}.")

        self.lagtime = lagtime

    @property
    def transition(self):
        return self.model.q_Y_given_X

    def _add_default_conditional_dist_params(
        self, cond_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if cond_params is None:
            cond_params = {}

        if not ("hidden_dims" in cond_params):
            cond_params["hidden_dims"] = [128]

        if not ("transform_name" in cond_params):
            cond_params["transform_name"] = "conditional_spline_autoregressive"

        if not ("n_transforms" in cond_params):
            cond_params["n_transforms"] = 2

        return cond_params

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

    def _instantiate_model(self, *args, **kwargs) -> tmm.bottleneck.TIB:
        self.model_params["mi_estimator"]["x_dim"] = self.n_dim
        self.model_params["mi_estimator"]["y_dim"] = self.n_dim
        self.model_params["mi_estimator"] = instantiate_estimator(
            **self.model_params["mi_estimator"]
        )

        q_Zt2_given_Zt1 = conditional_transformed_normal(
            input_dim=self.n_dim,
            context_dim=self.n_dim,
            **self.conditional_dist_params,
        )

        assert isinstance(self.proj, ConditionalDistribution)

        return tmm.bottleneck.TIB(
            q_Zt2_given_Zt1=q_Zt2_given_Zt1,
            p_Zt_given_Xt=self.proj,
            beta=self.beta,
            **self.model_params,
        )

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        **train_params,
    ):
        XT1 = X[self.lagtime :]
        XT2 = X[: -self.lagtime]

        return super().fit(XT1, XT2, **train_params)
