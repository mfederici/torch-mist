from typing import Optional, Union, Tuple, Dict, Callable, List, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset

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
    dataloader, _, _ = make_default_dataloaders(
        data=data,
        valid_data=None,
        batch_size=1,
        valid_percentage=0,
        test_percentage=0,
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


def is_valid_entry(
    entry: Union[Dict[str, Any], np.ndarray, torch.Tensor, Tuple[Any, ...]]
) -> bool:
    if isinstance(entry, torch.Tensor):
        return torch.sum(entry != entry) == 0
    elif isinstance(entry, np.ndarray):
        return np.sum(entry != entry) == 0
    elif hasattr(entry, "values"):
        is_valid = True
        for element in entry.values():
            if not is_valid_entry(element):
                is_valid = False
                break
        return is_valid
    elif isinstance(entry, tuple):
        is_valid = True
        for element in entry:
            if not is_valid_entry(element):
                is_valid = False
                break
        return is_valid
    else:
        return True


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


def filter_dataset(dataset: Dataset):
    # Remove invalid entries
    valid_ids = []
    for idx, entry in enumerate(dataset):
        if is_valid_entry(entry):
            valid_ids.append(idx)
    if len(valid_ids) != len(dataset):
        print(
            f"[Warning]: Removing {len(dataset)-len(valid_ids)} entries from the dataset"
        )
        dataset = Subset(dataset, valid_ids)

    return dataset


def is_data_loader(data: TensorDictLike):
    return isinstance(data, DataLoader)


def make_default_dataloader(
    data: TensorDictLike,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    filter_invalid_data: bool = True,
) -> DataLoader:
    if is_data_loader(data):
        dataloader = data
    else:
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")
        dataset = make_dataset(data)

        if filter_invalid_data:
            dataset = filter_dataset(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
    return dataloader


def make_dataloader(
    dataset: Dataset,
    dataloader: DataLoader,
    batch_size: int,
    num_workers: int,
) -> Optional[DataLoader]:
    if dataloader is None:
        if dataset is None:
            dataloader = None
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
    return dataloader


def make_splits(
    dataset: Dataset, valid_percentage: float, test_percentage: float
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    # Make a random train/valid/test split
    n_valid = int(len(dataset) * valid_percentage)
    n_test = int(len(dataset) * test_percentage)

    train_set, valid_set, test_set = random_split(
        dataset, [len(dataset) - n_valid - n_test, n_valid, n_test]
    )
    if n_valid == 0:
        valid_set = None
    if n_test == 0:
        test_set = None

    return train_set, valid_set, test_set


def parse_data(
    data: Optional[Union[Dataset, DataLoader]], filter: bool
) -> Tuple[Optional[Dataset], Optional[DataLoader]]:
    if data is None:
        dataset = None
        dataloader = None
    elif is_data_loader(data):
        dataloader = data
        dataset = None
    else:
        dataset = make_dataset(data)
        dataloader = None

    if dataset and filter:
        dataset = filter_dataset(dataset)

    return dataset, dataloader


def make_default_dataloaders(
    data: TensorDictLike,
    valid_data: Optional[TensorDictLike] = None,
    test_data: Optional[TensorDictLike] = None,
    valid_percentage: float = 0.1,
    test_percentage: float = 0.0,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    filter_invalid_data: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    if valid_percentage > 0 and not (valid_data is None):
        print(
            "[Warning]: valid_percentage will be ignored since valid_data is provided"
        )
        valid_percentage = 0

    if test_percentage > 0 and not (test_data is None):
        print(
            "[Warning]: test_percentage will be ignored since valid_data is provided"
        )
        test_percentage = 0

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

    # Use the same batch size as train if not specified
    if eval_batch_size is None:
        eval_batch_size = batch_size

    # Filter out NaNs
    if filter_invalid_data:
        train_set = filter_dataset(train_set)

    # Parse the data to distinguish dataset and dataloader
    valid_set, valid_loader = parse_data(valid_data, filter_invalid_data)
    test_set, test_loader = parse_data(test_data, filter_invalid_data)

    # Create splits
    if test_percentage + valid_percentage > 0:
        train_set, valid_set, test_set = make_splits(
            train_set, valid_percentage, test_percentage
        )
        train_loader = None

    train_loader = make_dataloader(
        dataset=train_set,
        dataloader=train_loader,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    valid_loader = make_dataloader(
        dataset=valid_set,
        dataloader=valid_loader,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = make_dataloader(
        dataset=test_set,
        dataloader=test_loader,
        batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    assert is_data_loader(train_loader)
    assert is_data_loader(valid_loader) or (valid_loader is None)
    assert is_data_loader(test_loader) or (test_loader is None)

    return train_loader, valid_loader, test_loader


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
