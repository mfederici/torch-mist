from typing import Optional, Union, Tuple, Dict, Callable, List, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Dataset

from torch_mist.estimators import MIEstimator, TransformedMIEstimator
from torch_mist.estimators.hybrid import PQHybridMIEstimator
from torch_mist.utils.data import SampleDataset, SameAttributeDataLoader
from torch_mist.utils.data.dataset import DataFrameDataset
from torch_mist.utils.data.loader import sample_same_value


TensorDictLike = Union[
    Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]],
    Dict[str, Union[torch.Tensor, np.ndarray]],
    Dataset,
    DataLoader,
    pd.DataFrame,
]


def prepare_variables(
    variables: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    v_args, v_kwargs = [], {}
    if torch.is_tensor(variables):
        variables = [variables]
    if isinstance(variables, dict):
        v_kwargs = {k: v.to(device) for k, v in variables.items()}
    elif isinstance(variables, tuple) or isinstance(variables, list):
        v_args = [v.to(device) for v in variables]

    return v_args, v_kwargs


def infer_dims(
    data: TensorDictLike,
) -> Dict[str, int]:
    dataloader, _ = make_default_dataloaders(
        data=data,
        valid_data=None,
        batch_size=1,
        valid_percentage=0,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    if not isinstance(batch, dict):
        assert len(batch) == 2
        batch = {"x": batch[0], "y": batch[1]}

    return {k: v.shape[-1] for k, v in batch.items()}


def convert_to_tensor(
    array: Union[torch.Tensor, np.array]
) -> torch.FloatTensor:
    if isinstance(array, np.ndarray):
        array = torch.FloatTensor(array)
    return array


def make_dataset(data: TensorDictLike) -> Dataset:
    # Validate the input combinations
    if isinstance(data, list) or isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("data should be a tuple of two elements (x, y).")
        x, y = data[0], data[1]

        x = convert_to_tensor(x)
        y = convert_to_tensor(y)

        # Unsqueeze an empty dimension if necesary
        if len(x.shape) == 1:
            x = x.view(-1, 1)
        if len(y.shape) == 1:
            y = y.view(-1, 1)

        dataset = SampleDataset({"x": x, "y": y})
    elif isinstance(data, dict):
        dataset = SampleDataset(data)
    elif isinstance(data, pd.DataFrame):
        dataset = DataFrameDataset(data)
    else:
        dataset = data

    return dataset


def is_data_loader(data: TensorDictLike):
    return isinstance(data, DataLoader)


def make_default_dataloaders(
    data: TensorDictLike,
    valid_data: Optional[TensorDictLike] = None,
    valid_percentage: float = 0.1,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    # Make the datasets if necessary
    if is_data_loader(data):
        train_loader = data
        train_set = data.dataset
        batch_size = data.batch_size
        num_workers = data.num_workers
    else:
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")
        train_set = make_dataset(data)
        train_loader = None

    if valid_data is None:
        valid_set = None
        valid_loader = None
    elif is_data_loader(valid_data):
        valid_loader = data
        valid_set = None
    else:
        valid_set = make_dataset(valid_data)
        valid_loader = None

    # Create a validation set if specified
    if valid_percentage > 0:
        if valid_set is None and valid_loader is None:
            # Make a random train/valid split
            n_valid = int(len(train_set) * valid_percentage)
            train_set, valid_set = random_split(
                train_set, [len(train_set) - n_valid, n_valid]
            )
            train_loader = None
        else:
            print(
                "[Warning]: valid_percentage is ignored since valid_set or valid_loader are already specified."
            )

    if train_loader is None:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    if valid_loader is None and not (valid_set is None):
        valid_loader = DataLoader(
            valid_set, batch_size=batch_size, num_workers=num_workers
        )

    assert is_data_loader(train_loader)
    assert is_data_loader(valid_loader) or (valid_loader is None)

    return train_loader, valid_loader


def update_dataloader(
    estimator: MIEstimator,
    dataloader: DataLoader,
) -> DataLoader:
    # TODO: this is kinda hacky.. find a better way

    # Check if we need to modify the dataloader to sample the same attributes
    _estimator = estimator
    transforms = []
    while isinstance(_estimator, TransformedMIEstimator):
        transforms.append(_estimator.transforms)
        _estimator = estimator.base_estimator

    # If required, change the data-loader to sample batches with the same attribute only
    if isinstance(_estimator, PQHybridMIEstimator):
        neg_samples = _estimator.neg_samples
        device = next(iter(estimator.parameters())).device
        transforms.append({"y->y": _estimator.quantize_y})

        def compute_attributes(samples) -> torch.Tensor:
            v_args, v_kwargs = prepare_variables(samples, device)

            if "y" in v_kwargs:
                y = v_kwargs["y"]
            else:
                assert len(v_args) == 2
                y = v_args[1]

            for transform in transforms:
                y = transform["y->y"](y)
            return y.data.cpu()

        if not isinstance(dataloader, SameAttributeDataLoader):
            dataloader = sample_same_value(
                dataloader, compute_attributes, neg_samples=neg_samples
            )

    return dataloader
