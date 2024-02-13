from typing import Optional, Union, Tuple, Dict, Callable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Dataset

from torch_mist.estimators import MIEstimator, TransformedMIEstimator
from torch_mist.estimators.hybrid import PQHybridMIEstimator
from torch_mist.utils.batch import unfold_samples, move_to_device
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


def make_default_dataloaders(
    data: TensorDictLike,
    valid_data: Optional[TensorDictLike] = None,
    valid_percentage: float = 0.1,
    batch_size: Optional[int] = None,
    num_workers: int = -1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    # Make the datasets if necessary
    if isinstance(data, DataLoader):
        train_loader = data
        train_set = None
    else:
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")
        train_set = make_dataset(data)
        train_loader = None

    if valid_data is None:
        valid_set = None
        valid_loader = None
    elif isinstance(valid_data, DataLoader):
        valid_loader = data
        valid_set = None
    else:
        valid_set = make_dataset(valid_data)
        valid_loader = None

    # Create a validation set if specified
    if valid_percentage > 0:
        if valid_set is None and valid_loader is None:
            if train_set is None:
                raise ValueError(
                    "Please use a tuple (x,y), a dictionary {'x':..,'y':...} or a Dataset instead of a DataLoader"
                    + " for valid_percentage>0. Alternatively, set valid_percentage=0 or specify valid_data."
                )
            else:
                # Make a random train/valid split
                n_valid = int(len(train_set) * valid_percentage)
                train_set, valid_set = random_split(
                    train_set, [len(train_set) - n_valid, n_valid]
                )
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

    assert isinstance(train_loader, DataLoader) and not (train_loader is None)
    assert isinstance(valid_loader, DataLoader) or (valid_loader is None)

    return train_loader, valid_loader


def modify_data_loader(
    data_loader: DataLoader,
    neg_samples: int,
    device: torch.device,
    transforms: List[Dict[str, Callable]],
) -> DataLoader:
    def compute_attributes(samples) -> torch.Tensor:
        variables = unfold_samples(samples)
        variables = move_to_device(variables, device)
        assert "y" in variables
        y = variables["y"]
        for transform in transforms:
            y = transform["y->y"](y)
        return y.data.cpu()

    if not isinstance(data_loader, SameAttributeDataLoader):
        data_loader = sample_same_value(
            data_loader, compute_attributes, neg_samples=neg_samples
        )

    return data_loader


def make_dataloaders(
    estimator: MIEstimator,
    data: TensorDictLike,
    valid_data: Optional[TensorDictLike] = None,
    valid_percentage: float = 0.1,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    train_loader, valid_loader = make_default_dataloaders(
        data=data,
        valid_data=valid_data,
        valid_percentage=valid_percentage,
        batch_size=batch_size,
        num_workers=num_workers,
    )

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
        train_loader = modify_data_loader(
            train_loader,
            transforms=transforms,
            device=device,
            neg_samples=neg_samples,
        )

        if valid_loader:
            valid_loader = modify_data_loader(
                valid_loader,
                transforms=transforms,
                device=device,
                neg_samples=neg_samples,
            )

    return train_loader, valid_loader
