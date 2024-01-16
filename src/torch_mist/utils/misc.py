import inspect
from typing import Optional, Any, Union, Tuple, Dict

import torch
from torch.utils.data import DataLoader, random_split

from torch_mist.estimators import MIEstimator, TransformedMIEstimator
from torch_mist.estimators.hybrid import PQHybridMIEstimator
from torch_mist.utils.batch import unfold_samples, move_to_device
from torch_mist.utils.data import SampleDataset, SameAttributeDataLoader
from torch_mist.utils.data.loader import sample_same_value


def _instantiate_dataloaders(
    estimator: MIEstimator,
    x: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    train_loader: Optional[Any] = None,
    valid_loader: Optional[Any] = None,
    valid_percentage: float = 0.1,
    batch_size: Optional[int] = None,
    num_workers: int = 8,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if (x is None) != (y is None):
        raise ValueError(
            "Either both x and y need to be specified or neither."
        )

    if not ((x is None) ^ (train_loader is None)):
        raise ValueError(
            "Either both x and y or the train_loader need to be specified."
        )
    if not (x is None):
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")

        # Make the dataloaders from the samples
        dataset = SampleDataset({"x": x, "y": y})
        if valid_percentage > 0:
            if not (valid_loader is None):
                raise ValueError(
                    "The valid_loader can't be specified when x and y are used"
                )

            n_valid = int(len(dataset) * valid_percentage)
            dataset, val_set = random_split(
                dataset, [len(dataset) - n_valid, n_valid]
            )
            valid_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
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

        def compute_attributes(samples):
            variables = unfold_samples(samples)
            variables = move_to_device(variables, device)
            assert "y" in variables
            y = variables["y"]
            for transform in transforms:
                y = transform["y->y"](y)
            return _estimator.generative_estimator.transforms["y->y"](
                y
            ).data.cpu()

        if not isinstance(train_loader, SameAttributeDataLoader):
            train_loader = sample_same_value(
                train_loader, compute_attributes, neg_samples=neg_samples
            )

        if not isinstance(valid_loader, SameAttributeDataLoader) and not (
            valid_loader is None
        ):
            valid_loader = sample_same_value(
                valid_loader, compute_attributes, neg_samples=neg_samples
            )

    return train_loader, valid_loader
