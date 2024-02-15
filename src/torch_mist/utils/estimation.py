import inspect
from typing import Optional, Any, Union, Type, Dict, Tuple, List, Callable

import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset

from torch_mist.estimators import MultiMIEstimator
from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.factories import instantiate_estimator
from torch_mist.utils.data.utils import infer_dims, TensorDictLike
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.logger.base import Logger, DummyLogger
from torch_mist.utils.train.mi_estimator import train_mi_estimator
from torch_mist.utils.evaluation import evaluate_mi


def _instantiate_estimator(
    instantiation_func: Callable[[Any], MIEstimator],
    data: TensorDictLike,
    x_key: Optional[str] = None,
    y_key: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> MIEstimator:
    dims = infer_dims(data)
    if x_key is None:
        x_key = "x"
    if y_key is None:
        y_key = "y"

    if x_key in dims:
        x_dim = dims[x_key]
    else:
        raise ValueError(
            "The data does not contain a key for 'x'.\n"
            + f"Please specify a value for x_key among {dims.keys()}"
        )

    if y_key in dims:
        y_dim = dims[y_key]
    else:
        raise ValueError(
            "The data does not contain a key for 'y'.\n"
            + f"Please specify a value for y_key among {dims.keys()}"
        )

    if "x_dim" in inspect.signature(instantiation_func).parameters:
        kwargs["x_dim"] = x_dim
    if "y_dim" in inspect.signature(instantiation_func).parameters:
        kwargs["y_dim"] = y_dim

    if verbose:
        print(f"Instantiating the estimator with {kwargs}")

    estimator = instantiation_func(
        **kwargs,
    )

    if verbose:
        print(estimator)

    return estimator


def estimate_mi(
    data: TensorDictLike,
    estimator_name: Optional[str] = None,
    estimator: Optional[MIEstimator] = None,
    valid_data: Optional[TensorDictLike] = None,
    test_data: Optional[TensorDictLike] = None,
    valid_percentage: float = 0.1,
    device: Union[torch.device, str] = torch.device("cpu"),
    max_epochs: Optional[int] = None,
    max_iterations: Optional[int] = None,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    logger: Optional[Union[Logger, bool]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0.2,
    batch_size: Optional[int] = 128,
    evaluation_batch_size: Optional[int] = None,
    num_workers: int = 0,
    early_stopping: bool = True,
    patience: int = 5,
    tolerance: float = 0.001,
    return_estimator: bool = False,
    fast_train: bool = False,
    hidden_dims: List[int] = [128, 64],
    x_key: str = "x",
    y_key: str = "y",
    **kwargs,
) -> Union[
    float,
    Tuple[float, pd.DataFrame],
    Tuple[float, MIEstimator],
    Tuple[float, MIEstimator, pd.DataFrame],
]:
    if max_epochs is None and max_iterations is None:
        if verbose:
            print(
                "max_epochs and max_iterations are not specified, using max_epochs=10 by default."
            )
        max_epochs = 10

    if estimator is None and estimator_name is None:
        raise ValueError(
            "Please specify a value for estimator or estimator_name."
        )

    if estimator is None:
        # Instantiate the estimator while inferring the size for x and y
        if verbose:
            print(f"Instantiating the {estimator_name} estimator")

        estimator = _instantiate_estimator(
            instantiation_func=instantiate_estimator,
            data=data,
            x_key=x_key,
            y_key=y_key,
            verbose=verbose,
            estimator_name=estimator_name,
            hidden_dims=hidden_dims,
            **kwargs,
        )

    # If using different key instead of 'x' and 'y'
    if x_key != "x" or y_key != "y":
        if not isinstance(estimator, MultiMIEstimator):
            estimator = MultiMIEstimator({(x_key, y_key): estimator})
        else:
            assert (x_key, y_key) in estimator.estimators

    if verbose:
        print("Training the estimator")

    if logger is None:
        logger = PandasLogger()
    elif logger is False:
        logger = DummyLogger()

    train_log = train_mi_estimator(
        estimator=estimator,
        data=data,
        valid_data=valid_data,
        valid_percentage=valid_percentage,
        device=device,
        max_epochs=max_epochs,
        max_iterations=max_iterations,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        verbose=verbose,
        logger=logger,
        lr_annealing=lr_annealing,
        warmup_percentage=warmup_percentage,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        num_workers=num_workers,
        fast_train=fast_train,
    )

    if verbose:
        print("Evaluating the value of Mutual Information")

    if test_data is None:
        print(
            "[Warning]: using the train_data to estimate the value of mutual information. Please specify test_data."
        )
        test_data = data

    if evaluation_batch_size is None:
        evaluation_batch_size = batch_size

    with logger.test():
        mi_value = evaluate_mi(
            estimator=estimator,
            data=test_data,
            batch_size=evaluation_batch_size,
            device=device,
            num_workers=num_workers,
        )

    if not (train_log is None) and return_estimator:
        return mi_value, estimator, train_log
    elif not (train_log is None):
        return mi_value, train_log
    elif return_estimator:
        return mi_value, estimator
    else:
        return mi_value
