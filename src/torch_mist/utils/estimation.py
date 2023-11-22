from typing import Optional, Any, Union, Type, Dict, Tuple

import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer, Adam

from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.factories import instantiate_estimator
from torch_mist.utils.train.mi_estimator import train_mi_estimator
from torch_mist.utils.evaluation import evaluate_mi


def estimate_mi(
    estimator_name: str,
    x: Optional[Union[torch.Tensor, np.array]] = None,
    y: Optional[Union[torch.Tensor, np.array]] = None,
    train_loader: Optional[Any] = None,
    valid_loader: Optional[Any] = None,
    test_loader: Optional[Any] = None,
    valid_percentage: float = 0.2,
    device: Union[torch.device, str] = torch.device("cpu"),
    max_epochs: int = 10,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    return_log: bool = True,
    lr_annealing: bool = False,
    warmup_percentage: float = 0.2,
    batch_size: Optional[int] = 64,
    evaluation_batch_size: Optional[int] = None,
    num_workers: int = 8,
    early_stopping: bool = True,
    patience: int = 3,
    delta: float = 0.001,
    return_estimator: bool = False,
    **kwargs,
) -> Union[
    float,
    Tuple[float, pd.DataFrame],
    Tuple[float, MIEstimator],
    Tuple[float, MIEstimator, pd.DataFrame],
]:
    if not (x is None):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
    if not (y is None):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

    if "x_dim" not in kwargs and not (x is None):
        kwargs["x_dim"] = x.shape[-1]
    if "y_dim" not in kwargs and not (y is None):
        kwargs["y_dim"] = y.shape[-1]

    if verbose:
        print(f"Instantiating the {estimator_name} estimator")
    estimator = instantiate_estimator(estimator_name=estimator_name, **kwargs)
    if verbose:
        print(estimator)
        print("Training the estimator")

    if evaluation_batch_size is None:
        evaluation_batch_size = batch_size

    train_log = train_mi_estimator(
        estimator=estimator,
        x=x,
        y=y,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_percentage=valid_percentage,
        device=device,
        max_epochs=max_epochs,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        verbose=verbose,
        return_log=return_log,
        lr_annealing=lr_annealing,
        warmup_percentage=warmup_percentage,
        batch_size=batch_size,
        early_stopping=early_stopping,
        patience=patience,
        delta=delta,
        num_workers=num_workers,
    )

    if verbose:
        print("Evaluating the value of Mutual Information")
    if not (test_loader is None):
        x = None
        y = None
    elif x is None:
        print(
            "Warning: using the train_loader to estimate the value of mutual information. Please specify a test_loader"
        )
        test_loader = train_loader

    mi_value = evaluate_mi(
        estimator=estimator,
        x=x,
        y=y,
        dataloader=test_loader,
        batch_size=evaluation_batch_size,
        device=device,
        num_workers=num_workers,
    )

    if return_log and return_estimator:
        return mi_value, estimator, train_log
    elif return_log:
        return mi_value, train_log
    elif return_estimator:
        return mi_value, estimator
    else:
        return mi_value
