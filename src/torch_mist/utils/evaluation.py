import collections
from typing import Optional, Any, List, Union, Dict

import torch
import numpy as np
from torch.utils.data import DataLoader

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.batch import unfold_samples, move_to_device
from torch_mist.utils.data.dataset import SampleDataset


def evaluate_mi(
    estimator: MIEstimator,
    x: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    dataloader: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    batch_size: Optional[int] = None,
    num_workers: int = 8,
) -> Union[float, Dict[str, float]]:
    if (x is None) != (y is None):
        raise ValueError(
            "Either both x and y need to be specified or neither."
        )
    if (
        (x is None and y is None and dataloader is None)
        or not (x is None)
        and not (y is None)
        and not (dataloader is None)
    ):
        raise ValueError(
            "Either both x and y or the train_loader need to be specified."
        )
    if not (x is None) and not (y is None):
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")

        # Make a train_loader from the samples
        dataset = SampleDataset({"x": x, "y": y})
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers
        )

    estimator.eval()
    estimator = estimator.to(device)

    mis = collections.defaultdict(list)
    unfold = False
    for samples in dataloader:
        variables = unfold_samples(samples)
        variables = move_to_device(variables, device)

        estimation = estimator(**variables)

        if isinstance(estimation, dict):
            for (x_key, y_key), value in estimation.items():
                mis[f"I({x_key};{y_key})"].append(value.item())
        else:
            mis["I(x;y)"].append(estimation.item())
            unfold = True

    mi = {}
    for name, values in mis.items():
        mi[name] = np.mean(values)

    if unfold:
        return mi["I(x;y)"]
    return mi
