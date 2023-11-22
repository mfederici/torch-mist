from typing import Optional, Any

import torch
import numpy as np
from torch.utils.data import DataLoader

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.batch import unfold_samples
from torch_mist.utils.data.dataset import SampleDataset


def evaluate_mi(
    estimator: MIEstimator,
    x: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    dataloader: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    batch_size: Optional[int] = None,
    num_workers: int = 8,
) -> float:
    mis = []

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

    for samples in dataloader:
        x, y = unfold_samples(samples)

        x = x.to(device)
        y = y.to(device)

        estimation = estimator(x, y)
        mis.append(estimation.item())

    return np.mean(mis)
