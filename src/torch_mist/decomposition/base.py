from abc import abstractmethod
from typing import Optional, Dict, Any, Union

import numpy as np
from pyro.distributions import ConditionalDistribution
from sklearn.base import TransformerMixin
import torch
from torch import nn

from torch_mist.distributions import conditional_transformed_normal
from torch_mist.estimators import (
    MIEstimator,
    instantiate_estimator,
    TransformedMIEstimator,
)
from torch_mist.nn import dense_nn, Model
from torch_mist.utils import train_mi_estimator
from torch_mist.utils.data.utils import TensorDictLike
from torch_mist.utils.train import train_model

DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_BATCH_SIZE = 64


class CenterAndScale:
    def __init__(
        self,
        loc: Union[torch.Tensor, np.ndarray],
        scale: Union[torch.Tensor, np.ndarray],
        min_scale=1e-6,
    ):
        super().__init__()
        self.loc = loc
        scale[np.abs(scale) < min_scale] = 1.0
        self.scale = scale

    def __call__(self, data):
        return (data - self.loc) / self.scale


class DimensionalityReduction(TransformerMixin):
    def __init__(
        self,
        n_dim: int,
        normalize_inputs: bool = True,
        whiten: bool = False,
        proj: Optional[nn.Module] = None,
        y_proj: Optional[nn.Module] = None,
        model: Optional[Model] = None,
        proj_params: Optional[Dict[str, Any]] = None,
        y_proj_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.proj = proj
        self.y_proj = y_proj
        self.model = model

        self.proj_params = self._add_default_proj_params(proj_params)
        self.y_proj_params = self._add_default_y_proj_params(y_proj_params)
        self.model_params = self._add_default_model_params(model_params)

        self.normalize_inputs = normalize_inputs
        self.normalize_projection = whiten
        self.normalize_X = None
        self.normalize_Y = None
        self.normalize_Z = None
        self.train_log = None

    def _add_default_model_params(
        self, model_params: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        return model_params

    def _add_default_proj_params(
        self, proj_params: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        return proj_params

    def _add_default_y_proj_params(
        self, y_proj_params: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        return y_proj_params

    def _add_default_train_params(
        self, train_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if train_params is None:
            train_params = {}

        if not ("batch_size" in train_params):
            print(
                "[Info]: batch_size is not specified,"
                + f" using batch_size={DEFAULT_BATCH_SIZE} by default."
            )
            train_params["batch_size"] = DEFAULT_BATCH_SIZE

        if not ("max_epochs" in train_params) and not (
            "max_iterations" in train_params
        ):
            print(
                "[Info]: max_epoch and max_iterations are not specified,"
                + f" using max_iterations={DEFAULT_MAX_ITERATIONS} by default."
            )
            train_params["max_iterations"] = DEFAULT_MAX_ITERATIONS

        if not ("early_stopping" in train_params):
            train_params["early_stopping"] = True

        if not ("verbose" in train_params):
            train_params["verbose"] = False

        return train_params

    @abstractmethod
    def _instantiate_proj(self, x_dim: int) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def _instantiate_y_proj(self, y_dim: int) -> Optional[nn.Module]:
        raise NotImplementedError()

    @abstractmethod
    def _instantiate_model(self, x_dim: int, y_dim: int) -> Model:
        raise NotImplementedError()

    def _train_model(self, data: TensorDictLike, **train_params):
        return train_model(
            self.model, data, eval_method="loss", **train_params
        )

    def _get_transformed_y_dim(self, Y: np.ndarray) -> int:
        if self.y_proj is None:
            y_dim = Y.shape[-1]
        else:
            y_dim = self.y_proj(torch.FloatTensor(Y[:2])).shape[-1]
        return y_dim

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        **train_params,
    ):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        x_dim = X.shape[-1]
        y_dim = self._get_transformed_y_dim(Y)

        if self.normalize_inputs:
            self.normalize_X = CenterAndScale(
                X.mean(0, keepdims=True), X.std(0, keepdims=True)
            )
            CX = self.normalize_X(X)
            self.normalize_Y = CenterAndScale(
                Y.mean(0, keepdims=True), Y.std(0, keepdims=True)
            )
            CY = self.normalize_Y(Y)
        else:
            CX = X
            CY = Y

        if self.proj is None:
            self.proj = self._instantiate_proj(x_dim)
        if self.y_proj is None:
            self.y_proj = self._instantiate_y_proj(y_dim)
        if self.model is None:
            self.model = self._instantiate_model(x_dim=x_dim, y_dim=y_dim)

        train_params = self._add_default_train_params(train_params)

        self.train_log = self._train_model(data=(CX, CY), **train_params)

        if self.normalize_projection:
            self.normalize_Z = None
            Z = self.transform(X)
            self.normalize_Z = CenterAndScale(
                Z.mean(0, keepdims=True), Z.std(0, keepdims=True)
            )

        return self

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.proj(X)

    def transform(
        self,
        X: np.ndarray,
    ) -> Union[np.ndarray, torch.Tensor]:
        if self.proj is None:
            raise ValueError(
                "Use the fit(X, Y) method before calling transform(X)."
            )
        self.proj.eval()

        if self.normalize_X:
            X = self.normalize_X(X)

        is_numpy = False
        device = next(iter(self.proj.parameters())).device

        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            is_numpy = True

        original_device = X.device
        with torch.no_grad():
            z = self._encode(X.to(device)).to(original_device)
            if not (self.normalize_Z is None):
                z = self.normalize_Z(z)

        if is_numpy:
            z = z.data.numpy()

        return z


class StochasticDimensionalityReduction(DimensionalityReduction):
    def __init__(self, *args, stochastic_transform: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stochastic_transform = stochastic_transform

    def _instantiate_proj(self, x_dim: int) -> nn.Module:
        self.proj_params["input_dim"] = self.n_dim
        self.proj_params["context_dim"] = x_dim
        return conditional_transformed_normal(**self.proj_params)

    def _get_transformed_y_dim(self, Y: np.ndarray) -> int:
        if self.y_proj is None:
            y_dim = Y.shape[-1]
        elif isinstance(self.y_proj, ConditionalDistribution):
            y_dim = (
                self.y_proj.condition(torch.FloatTensor(Y[:2]))
                .sample()
                .shape[-1]
            )
        else:
            y_dim = self.y_proj(torch.FloatTensor(Y[:2])).shape[-1]
        return y_dim

    def _encode(self, X: torch.Tensor) -> torch.Tensor:
        p_Z_given_x = self.proj.condition(X)
        if self.stochastic_transform:
            return p_Z_given_x.sample()
        else:
            z = torch.zeros(*X.shape[:-1], self.n_dim).to(X.device)
            # Transform the mean of the base distribution (which is also the mode for normals).
            z += p_Z_given_x.base_dist.mean
            for transform in p_Z_given_x.transforms:
                z = transform._call(z)
            return z
