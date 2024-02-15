from typing import Callable, List, Optional, Dict, Any, Union

import numpy as np
from sklearn.base import TransformerMixin, ClassNamePrefixFeaturesOutMixin
import torch
from torch import nn

from torch_mist.estimators import (
    MIEstimator,
    instantiate_estimator,
    TransformedMIEstimator,
)
from torch_mist.nn import dense_nn
from torch_mist.utils import train_mi_estimator


class CenterAndScale(nn.Module):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("loc", torch.FloatTensor(loc))
        self.register_buffer("scale", torch.FloatTensor(scale))

    def forward(self, data):
        return (data - self.loc) / self.scale


class MID(TransformerMixin):
    def __init__(
        self,
        n_dim: int,
        normalize_inputs: bool = True,
        whiten: bool = False,
        proj: Optional[nn.Module] = None,
        y_proj: Optional[nn.Module] = None,
        mi_estimator: Optional[MIEstimator] = None,
        proj_params: Optional[Dict[str, Any]] = None,
        mi_estimator_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.n_dim = n_dim
        self._proj = proj
        self._y_proj = y_proj
        self._mi_estimator = mi_estimator

        if proj_params is None:
            proj_params = {}
        self.proj_params = self._add_default_proj_params(proj_params)
        if mi_estimator_params is None:
            mi_estimator_params = {}
        self.mi_estimator_params = self._add_default_mi_estimator_params(
            mi_estimator_params
        )

        self.normalize_inputs = normalize_inputs
        self.normalize_projection = whiten
        self.normalize_X = None
        self.normalize_Y = None
        self.normalize_Z = None
        self.train_log = None

    @property
    def proj(self):
        if self._proj is None:
            return None
        layers = [self._proj]

        if not (self.normalize_X is None):
            layers = [self.normalize_X] + layers
        return nn.Sequential(*layers)

    @property
    def y_proj(self):
        if self.normalize_Y is None:
            return self._y_proj
        else:
            if self._y_proj is None:
                return self.normalize_Y
            else:
                return nn.Sequential(self.normalize_Y, self._proj)

    @property
    def mi_estimator(self):
        if self._mi_estimator is None:
            return None

        transforms = {"x": self.proj}
        if not (self.y_proj is None):
            transforms["y"] = self.y_proj

        return TransformedMIEstimator(
            transforms=transforms, base_estimator=self._mi_estimator
        )

    def _add_default_mi_estimator_params(
        self, mi_estimator_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not ("hidden_dims" in mi_estimator_params):
            mi_estimator_params["hidden_dims"] = [128, 64]

        if not ("estimator_name" in mi_estimator_params):
            mi_estimator_params["estimator_name"] = "smile"
            if not ("neg_samples" in mi_estimator_params):
                mi_estimator_params["neg_samples"] = 8
        if not ("nonlinearity" in mi_estimator_params):
            mi_estimator_params["nonlinearity"] = nn.ReLU(True)

        mi_estimator_params["x_dim"] = self.n_dim

        return mi_estimator_params

    def _add_default_proj_params(
        self, proj_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not ("hidden_dims" in proj_params):
            proj_params["hidden_dims"] = [64]
        if not ("nonlinearity" in proj_params):
            proj_params["nonlinearity"] = nn.ReLU(True)

        proj_params["output_dim"] = self.n_dim
        return proj_params

    def _add_default_train_params(
        self, train_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not ("batch_size" in train_params):
            train_params["batch_size"] = 64

        if not ("max_epochs" in train_params) and not (
            "max_iterations" in train_params
        ):
            train_params["max_epochs"] = 10

        return train_params

    def _instantiate_proj(self, input_dim: int):
        self.proj_params["input_dim"] = input_dim
        self._proj = dense_nn(**self.proj_params)

    def _instantiate_mi_estimator(self, y_dim: int):
        self.mi_estimator_params["x_dim"] = self.n_dim
        self.mi_estimator_params["y_dim"] = y_dim
        self._mi_estimator = instantiate_estimator(**self.mi_estimator_params)

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        **train_params
    ):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        x_dim = X.shape[-1]
        y_dim = Y.shape[-1]

        # Adapt the dimension of y if we are projecting
        if not (self._y_proj is None):
            y_dim = self._y_proj(torch.FloatTensor(Y[:2])).shape[-1]

        if self.normalize_inputs:
            self.normalize_X = CenterAndScale(
                X.mean(0, keepdims=True), X.std(0, keepdims=True)
            )
            self.normalize_Y = CenterAndScale(
                Y.mean(0, keepdims=True), Y.std(0, keepdims=True)
            )

        if self._proj is None:
            self._instantiate_proj(input_dim=x_dim)
        if self._mi_estimator is None:
            self._instantiate_mi_estimator(y_dim=y_dim)

        train_params = self._add_default_train_params(train_params)

        self.train_log = train_mi_estimator(
            estimator=self.mi_estimator, data=(X, Y), **train_params
        )

        if self.normalize_projection:
            Z = self.transform(X)
            self.normalize_Z = CenterAndScale(
                Z.mean(0, keepdims=True), Z.std(0, keepdims=True)
            )

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.proj is None:
            raise ValueError(
                "Use the fit(X, Y) method before calling transform(X)."
            )
        self.proj.eval()

        is_numpy = False
        device = next(iter(self.proj.parameters())).device

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            is_numpy = True

        original_device = X.device
        with torch.no_grad():
            z = self.proj(X.to(device)).to(original_device)
            if not (self.normalize_Z is None):
                z = self.normalize_Z(z)

        if is_numpy:
            z = z.data.numpy()

        return z
