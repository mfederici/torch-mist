from typing import Optional, Any, Union, Type, Dict, Tuple, List

import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset

from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.factories import instantiate_estimator
from torch_mist.utils.batch import unfold_samples
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.logger.base import Logger, DummyLogger
from torch_mist.utils.misc import make_default_dataloaders
from torch_mist.utils.train.mi_estimator import train_mi_estimator
from torch_mist.utils.evaluation import evaluate_mi


def _infer_dim(
    data: Union[
        Tuple[
            Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]
        ],
        Dict[str, Union[torch.Tensor, np.ndarray]],
        Dataset,
        DataLoader,
    ],
) -> Tuple[int, int]:
    dataloader, _ = make_default_dataloaders(
        data=data,
        valid_data=None,
        batch_size=1,
        valid_percentage=0,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    variables = unfold_samples(batch)
    if not ("x" in variables):
        raise ValueError(
            "Each batch must consist of a tuple (x,y) or a dictionary containing a key for 'x'"
        )
    x_dim = variables["x"].shape[-1]
    if not ("y" in variables):
        raise ValueError(
            "Each batch must consist of a tuple (x,y) or a dictionary containing a key for 'y'"
        )
    y_dim = variables["y"].shape[-1]

    return x_dim, y_dim


def estimate_mi(
    estimator_name: str,
    data: Union[
        Tuple[
            Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]
        ],
        Dict[str, Union[torch.Tensor, np.ndarray]],
        Dataset,
        DataLoader,
    ],
    valid_data: Optional[
        Union[
            Tuple[
                Union[torch.Tensor, np.ndarray],
                Union[torch.Tensor, np.ndarray],
            ],
            Dict[str, Union[torch.Tensor, np.ndarray]],
            Dataset,
            DataLoader,
        ]
    ] = None,
    test_data: Optional[
        Union[
            Tuple[
                Union[torch.Tensor, np.ndarray],
                Union[torch.Tensor, np.ndarray],
            ],
            Dict[str, Union[torch.Tensor, np.ndarray]],
            Dataset,
            DataLoader,
        ]
    ] = None,
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
    patience: int = 10,
    delta: float = 0.001,
    return_estimator: bool = False,
    fast_train: bool = False,
    hidden_dims: List[int] = [128, 64],
    **kwargs,
) -> Union[
    float,
    Tuple[float, pd.DataFrame],
    Tuple[float, MIEstimator],
    Tuple[float, MIEstimator, pd.DataFrame],
]:
    x_dim, y_dim = _infer_dim(data)

    if max_epochs is None and max_iterations is None:
        if verbose:
            print(
                "max_epochs and max_iterations are not specified, using max_epochs=10 by default."
            )
        max_epochs = 10

    if verbose:
        print(f"Instantiating the {estimator_name} estimator")
    estimator = instantiate_estimator(
        estimator_name=estimator_name,
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        **kwargs,
    )
    if verbose:
        print(estimator)
        print("Training the estimator")

    if evaluation_batch_size is None:
        evaluation_batch_size = batch_size

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
        delta=delta,
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

    with logger.test():
        mi_value = evaluate_mi(
            estimator=estimator,
            data=test_data,
            batch_size=evaluation_batch_size,
            device=device,
            num_workers=num_workers,
        )

    logger.clear()

    if not (train_log is None) and return_estimator:
        return mi_value, estimator, train_log
    elif not (train_log is None):
        return mi_value, train_log
    elif return_estimator:
        return mi_value, estimator
    else:
        return mi_value
