import copy
import inspect
import random
from typing import Optional, Any, Union, Type, Dict, Tuple, List, Callable

import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from tqdm.auto import tqdm

from torch_mist.estimators import MultiMIEstimator
from torch_mist.estimators.base import MIEstimator
from torch_mist.estimators.factories import instantiate_estimator
from torch_mist.utils.data.utils import (
    infer_dims,
    TensorDictLike,
    make_dataset,
    is_data_loader,
    is_valid_entry,
)
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.logger.base import Logger, DummyLogger
from torch_mist.utils.train.mi_estimator import train_mi_estimator
from torch_mist.utils.evaluation import evaluate_mi

DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_MAX_EPOCHS = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_ESTIMATOR = "smile"


def _instantiate_estimator(
    estimator: Union[str, MIEstimator],
    data: TensorDictLike,
    instantiation_func: Optional[Callable[[Any], MIEstimator]] = None,
    x_key: Optional[str] = None,
    y_key: Optional[str] = None,
    verbose: bool = True,
    **estimator_params,
) -> MIEstimator:
    if isinstance(estimator, str):
        if instantiation_func is None:
            instantiation_func = instantiate_estimator
        # Instantiate the estimator while inferring the size for x and y
        if verbose:
            print(f"Instantiating the {estimator} estimator")

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
            estimator_params["x_dim"] = x_dim
        if "y_dim" in inspect.signature(instantiation_func).parameters:
            estimator_params["y_dim"] = y_dim

        if verbose:
            print(f"Instantiating the estimator with {estimator_params}")

        estimator_params["estimator_name"] = estimator
        estimator = instantiation_func(
            **estimator_params,
        )

    if not isinstance(estimator, MIEstimator):
        raise ValueError(
            f"Invalid estimator: {estimator}."
            + " estimator can be either an instance of MIEstimator or a string referring to a factory."
        )

    if verbose:
        print(estimator)

    return estimator


def _determine_train_duration(
    max_iterations: Optional[int],
    max_epochs: Optional[int],
    data: TensorDictLike,
) -> Tuple[Optional[int], Optional[int]]:
    if max_epochs is None and max_iterations is None:
        if is_data_loader(data):
            max_epochs = DEFAULT_MAX_EPOCHS
            print(
                f"[Info]: max_epochs and max_iterations are not specified, using max_epochs={max_epochs} by default."
            )
        else:
            max_iterations = DEFAULT_MAX_ITERATIONS
            print(
                f"[Info]: max_epochs and max_iterations are not specified, using max_iterations={max_iterations} by default."
            )
    return max_iterations, max_epochs


def estimate_mi(
    data: TensorDictLike,
    estimator: Union[MIEstimator, str] = DEFAULT_ESTIMATOR,
    valid_data: Optional[TensorDictLike] = None,
    test_data: Optional[Union[TensorDictLike, bool]] = None,
    valid_percentage: float = 0.1,
    device: Union[torch.device, str] = torch.device("cpu"),
    max_epochs: Optional[int] = None,
    max_iterations: Optional[int] = None,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    logger: Optional[Union[Logger, bool]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0,
    batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
    evaluation_batch_size: Optional[int] = None,
    num_workers: int = 0,
    early_stopping: bool = True,
    patience: Optional[int] = None,
    tolerance: float = 0.001,
    return_estimator: bool = False,
    fast_train: bool = False,
    x_key: str = "x",
    y_key: str = "y",
    **estimator_params,
) -> Union[
    float,
    MIEstimator,
    Tuple[float, pd.DataFrame],
    Tuple[float, MIEstimator],
    Tuple[float, MIEstimator, pd.DataFrame],
]:
    max_iterations, max_epochs = _determine_train_duration(
        max_iterations=max_iterations, max_epochs=max_epochs, data=data
    )

    estimator = _instantiate_estimator(
        estimator=estimator, data=data, verbose=verbose, **estimator_params
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
        train_data=data,
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
            "[Warning]: using data to estimate the value of mutual information. Please specify test_data."
        )
        test_data = data

    if evaluation_batch_size is None:
        evaluation_batch_size = batch_size

    out = []
    if not (test_data is False):
        with logger.test():
            with logger.logged_methods(
                estimator,
                methods=["mutual_information"],
            ):
                mi_value = evaluate_mi(
                    estimator=estimator,
                    data=test_data,
                    batch_size=evaluation_batch_size,
                    device=device,
                    num_workers=num_workers,
                )
        out.append(mi_value)

    if return_estimator:
        out.append(estimator)
    if not (train_log is None):
        out.append(train_log)

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def _train_on_fold(
    chunks: List[Dataset],
    fold: int,
    device: Union[str, torch.device],
    verbose: bool,
    batch_size: int,
    evaluation_batch_size: Optional[int] = None,
    num_workers: int = 0,
    **train_params,
) -> Tuple[Dict[str, float], int, int]:
    if evaluation_batch_size is None:
        evaluation_batch_size = batch_size

    test_fold = fold
    valid_fold = (test_fold + 1) % len(chunks)
    train_folds = [
        f for f in range(len(chunks)) if f != test_fold and f != valid_fold
    ]

    assert (
        test_fold != valid_fold
        and not (test_fold in train_folds)
        and not (valid_fold in train_folds)
    )

    datasets = {
        "train": ConcatDataset(
            [chunks[train_fold] for train_fold in train_folds]
        ),
        "valid": chunks[valid_fold],
        "test": chunks[test_fold],
        "all": ConcatDataset(chunks),
    }

    train_logger = DummyLogger()
    estimator = estimate_mi(
        data=datasets["train"],
        valid_data=datasets["valid"],
        test_data=False,
        valid_percentage=0,
        logger=train_logger,
        device=device,
        verbose=verbose,
        batch_size=batch_size,
        return_estimator=True,
        **train_params,
    )

    iterations = train_logger._iteration
    epochs = train_logger._epoch
    mi_values = {}

    # Evaluate on the splits
    for split, dataset in datasets.items():
        mi = evaluate_mi(
            estimator,
            data=dataset,
            batch_size=evaluation_batch_size,
            device=device,
            num_workers=num_workers,
        )
        if isinstance(mi, dict):
            if len(mi) > 1:
                raise ValueError(
                    "k_fold_mi_estimation is not supported when estimating multiple values of mutual information"
                )
            mi = next(iter(mi.values()))

        mi_values[split] = mi
    return mi_values, iterations, epochs


def _prepare_k_fold_data(
    data: TensorDictLike,
    folds: int,
    seed: Optional[int],
    verbose: bool,
) -> List[Dataset]:
    full_dataset = make_dataset(data)

    # Create k train-test splits
    if verbose:
        print(f"The dataset has {len(full_dataset)} entries.")
        print(f"Creating the {folds} train/validation/test splits")

    if not (seed is None):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Create a permutation
    ids_permutation = np.random.permutation(len(full_dataset))

    # Drop the last to make each split has the same size
    if len(full_dataset) % folds != 0:
        ids_permutation = ids_permutation[: -(len(full_dataset) % folds)]
        print(
            f"[Warning]: Dropping {len(full_dataset) % folds} entries to create even splits."
        )

    # Create 10 folds
    ids_folds = np.split(ids_permutation, folds)

    # Filter out the invalid ids (for NaN entries if any)
    ids_folds = [
        [idx for idx in ids_fold if is_valid_entry(full_dataset[idx])]
        for ids_fold in ids_folds
    ]

    chunks = [Subset(full_dataset, ids_fold) for ids_fold in ids_folds]

    total_valid_size = sum([len(chunk) for chunk in chunks])
    if total_valid_size < len(full_dataset):
        print(
            f"[Warning]: Removed {len(full_dataset)-total_valid_size} invalid entries."
        )

    return chunks


def k_fold_mi_estimate(
    data: TensorDictLike,
    estimator: Union[MIEstimator, str] = DEFAULT_ESTIMATOR,
    verbose: bool = True,
    verbose_train: bool = False,
    logger: Optional[Union[Logger, bool]] = None,
    seed: Optional[int] = None,
    folds: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Union[str, torch.device] = torch.device("cpu"),
    n_estimations: Optional[int] = None,
    **kwargs,
) -> Tuple[float, Any]:
    if isinstance(data, DataLoader):
        raise ValueError(
            "DataLoaders are not supported for k_fold_mi_estimate, please provide a Dataset instead."
        )

    if logger is None:
        logger = PandasLogger()
    elif logger is False:
        logger = DummyLogger()

    # Chunk the dataset
    chunks = _prepare_k_fold_data(
        data=data, folds=folds, verbose=verbose, seed=seed
    )

    results = []
    total_iterations = 0
    total_epochs = 0

    if n_estimations is None:
        n_estimations = folds

    if n_estimations > folds:
        raise ValueError(
            "The number of estimations has to be less than the number of folds n_estimations<=n_folds (default=n_folds)"
        )

    tqdm_fold = (
        tqdm(total=n_estimations, desc="Fold", position=1) if verbose else None
    )

    for fold in range(n_estimations):
        mi_values, iterations, epochs = _train_on_fold(
            chunks=chunks,
            estimator=estimator,
            fold=fold,
            device=device,
            batch_size=batch_size,
            verbose=verbose_train,
            **kwargs,
        )

        total_iterations += iterations
        total_epochs += epochs

        # log the results for the splits
        for split, mi in mi_values.items():
            logger._log(
                data=mi,
                name="mutual_information",
                split=split,
                iteration=total_iterations,
                epoch=total_epochs,
            )

        results.append(mi_values["test"])

        if tqdm_fold:
            tqdm_fold.set_postfix_str(
                f"mutual_information: {np.mean(results)} nats"
            )
            tqdm_fold.update(1)

    log = logger.get_log()

    return np.mean(results), log
