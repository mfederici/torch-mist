import functools
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
from torch_mist.estimators.temporal import TemporalMIEstimator
from torch_mist.utils.data.temporal import temporal_offset_data
from torch_mist.utils.data.utils import (
    infer_dims,
    TensorDictLike,
    make_dataset,
    is_data_loader,
    is_valid_entry,
    make_default_dataloaders,
)
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.logger.base import Logger, DummyLogger
from torch_mist.utils.train.mi_estimator import train_mi_estimator
from torch_mist.utils.evaluation import evaluate_mi

DEFAULTS = {
    "max_iterations": 10000,
    "max_epochs": 20,
    "batch_size": 128,
    "estimator_name": "smile",
}


def _instantiate_estimator(
    estimator: Union[str, MIEstimator, Callable[[Any], MIEstimator]],
    data: TensorDictLike,
    x_key: Optional[str] = None,
    y_key: Optional[str] = None,
    verbose: bool = True,
    **estimator_params,
) -> MIEstimator:
    if isinstance(estimator, MIEstimator):
        return estimator
    elif isinstance(estimator, str) or hasattr(estimator, "__call__"):
        if isinstance(estimator, str):
            instantiation_func = instantiate_estimator
            estimator_params["estimator_name"] = estimator
        else:
            assert isinstance(estimator, functools.partial)
            assert len(estimator.args) == 0
            instantiation_func = estimator.func
            estimator_params.update(estimator.keywords)

        # Instantiate the estimator while inferring the size for x and y
        dims = infer_dims(data)

        if x_key is None:
            x_key = "x"
        if y_key is None:
            y_key = "y"

        if verbose:
            print(f"The data has the following components: {dims}")
            print(f"Estimating mutual information between {x_key} and {y_key}")

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

        estimator_params["x_dim"] = x_dim
        estimator_params["y_dim"] = y_dim

        if verbose:
            newline = "\n  "
            print(
                f"Instantiating the {instantiation_func.__name__} estimator with \n  {newline.join([f'{k}={v}' for k,v in estimator_params.items()])}"
            )

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
        # Count and visualize the number of parameters
        n_parameters = 0
        for param in estimator.parameters():
            n_parameters += param.numel()
        print(f"{n_parameters} Parameters")

    # If using different key instead of 'x' and 'y'
    if x_key != "x" or y_key != "y":
        if not isinstance(estimator, MultiMIEstimator):
            estimator = MultiMIEstimator({(x_key, y_key): estimator})
        else:
            assert (x_key, y_key) in estimator.estimators

    return estimator


def _determine_train_duration(
    max_iterations: Optional[int],
    max_epochs: Optional[int],
    data: TensorDictLike,
) -> Tuple[Optional[int], Optional[int]]:
    if max_epochs is None and max_iterations is None:
        if is_data_loader(data):
            max_epochs = DEFAULTS["max_epochs"]
            print(
                f"[Info]: max_epochs and max_iterations are not specified, using max_epochs={max_epochs} by default."
            )
        else:
            max_iterations = DEFAULTS["max_iterations"]
            print(
                f"[Info]: max_epochs and max_iterations are not specified, using max_iterations={max_iterations} by default."
            )
    return max_iterations, max_epochs


def estimate_mi(
    data: TensorDictLike,
    estimator: Union[MIEstimator, str] = DEFAULTS["estimator_name"],
    valid_data: Optional[TensorDictLike] = None,
    test_data: Optional[Union[TensorDictLike, bool]] = None,
    valid_percentage: float = 0.1,
    test_percentage: float = 0.0,
    device: Union[torch.device, str] = torch.device("cpu"),
    max_epochs: Optional[int] = None,
    max_iterations: Optional[int] = None,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    logger: Optional[Union[Logger, bool]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0,
    batch_size: Optional[int] = DEFAULTS["batch_size"],
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    early_stopping: bool = True,
    patience: Optional[int] = None,
    tolerance: float = 0.001,
    return_estimator: bool = False,
    fast_train: bool = False,
    x_key: str = "x",
    y_key: str = "y",
    train_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    trained_model_save_path: Optional[str] = None,
    save_train_log: bool = True,
    **estimator_params,
) -> Union[
    Union[Dict[str, float], float],
    Tuple[Union[Dict[str, float], float], pd.DataFrame],
    Tuple[Union[Dict[str, float], float], MIEstimator],
    Tuple[Union[Dict[str, float], float], MIEstimator, pd.DataFrame],
]:
    max_iterations, max_epochs = _determine_train_duration(
        max_iterations=max_iterations, max_epochs=max_epochs, data=data
    )

    estimator = _instantiate_estimator(
        estimator=estimator,
        data=data,
        verbose=verbose,
        x_key=x_key,
        y_key=y_key,
        **estimator_params,
    )

    if verbose:
        print("Training the estimator")

    if logger is None:
        logger = PandasLogger()
    elif logger is False:
        logger = DummyLogger()

    if eval_batch_size is None:
        eval_batch_size = batch_size

    # Prepare the data
    train_loader, valid_loader, test_loader = make_default_dataloaders(
        data=data,
        valid_data=valid_data,
        test_data=None if test_data is False else test_data,
        valid_percentage=valid_percentage,
        test_percentage=test_percentage,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    if verbose:
        print(f"Train size: {len(train_loader.dataset)}")
        if not (valid_loader is None):
            print(f"Valid size: {len(valid_loader.dataset)}")
        if not (test_loader is None):
            print(f"Test size: {len(test_loader.dataset)}")

    train_log = train_mi_estimator(
        estimator=estimator,
        train_data=train_loader,
        valid_data=valid_loader,
        valid_percentage=0,
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
        train_logged_methods=train_logged_methods,
        eval_logged_methods=eval_logged_methods,
    )

    if verbose:
        print("Evaluating the value of Mutual Information")

    if not (test_data is False):
        if test_loader is None:
            print(
                "[Warning]: using data to estimate the value of mutual information. Please specify test_data or test_percentage>0."
            )
            test_data = data
        else:
            test_data = test_loader

        with logger.test():
            with logger.logged_methods(
                estimator,
                methods=["mutual_information"],
            ):
                mi_value = evaluate_mi(
                    estimator=estimator,
                    data=test_data,
                    batch_size=eval_batch_size,
                    device=device,
                    num_workers=num_workers,
                )
    else:
        mi_value = -1

    out = [mi_value]

    if save_train_log:
        logger.save_log()

    if not (trained_model_save_path is None):
        print(f"Saving the estimator in {trained_model_save_path}")
        logger.save_model(estimator, trained_model_save_path)

    if return_estimator:
        out.append(estimator)
    if not (train_log is None):
        out.append(train_log)

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def estimate_temporal_mi(
    data: Union[np.ndarray, torch.Tensor],
    lagtimes: Union[List[int], np.ndarray, torch.Tensor],
    estimator: Union[MIEstimator, str] = DEFAULTS["estimator_name"],
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
    batch_size: Optional[int] = DEFAULTS["batch_size"],
    evaluation_batch_size: Optional[int] = None,
    num_workers: int = 0,
    early_stopping: bool = True,
    patience: Optional[int] = None,
    tolerance: float = 0.001,
    return_estimator: bool = False,
    fast_train: bool = False,
    train_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    **estimator_params,
) -> Union[
    Dict[str, float],
    Tuple[Dict[str, float], pd.DataFrame],
    Tuple[Dict[str, float], MIEstimator],
    Tuple[Dict[str, float], MIEstimator, pd.DataFrame],
]:
    base_estimator = _instantiate_estimator(
        estimator=estimator,
        data={"x": data[:2], "y": data[:2]},
        verbose=verbose,
        **estimator_params,
    )

    estimator = TemporalMIEstimator(
        base_estimator=base_estimator, lagtimes=lagtimes
    )

    data = temporal_offset_data(data, lagtimes)
    if not (valid_data is None):
        valid_data = temporal_offset_data(valid_data, lagtimes)
    if not (test_data is None):
        test_data = temporal_offset_data(test_data, lagtimes)

    return estimate_mi(
        data=data,
        valid_data=valid_data,
        test_data=test_data,
        estimator=estimator,
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
        eval_batch_size=evaluation_batch_size,
        num_workers=num_workers,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        return_estimator=return_estimator,
        fast_train=fast_train,
        train_logged_methods=train_logged_methods,
        eval_logged_methods=eval_logged_methods,
    )


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
    _, estimator = estimate_mi(
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
    estimator: Union[MIEstimator, str] = DEFAULTS["estimator_name"],
    verbose: bool = True,
    verbose_train: bool = False,
    logger: Optional[Union[Logger, bool]] = None,
    seed: Optional[int] = None,
    folds: int = 10,
    batch_size: int = DEFAULTS["batch_size"],
    device: Union[str, torch.device] = torch.device("cpu"),
    n_estimations: Optional[int] = None,
    save_log: bool = True,
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
    mi = np.mean(results)
    if verbose:
        print(
            f"Mutual Information on test: {np.round(mi, 3)} +- {np.round(np.std(results),3)}"
        )
    if save_log:
        logger.save_log()

    return mi, log
