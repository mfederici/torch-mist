import collections
from typing import Optional, Any, List, Union, Dict, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.batch import unfold_samples, move_to_device
from torch_mist.utils.misc import make_dataloaders


def evaluate_mi(
    estimator: MIEstimator,
    data: Union[
        Tuple[
            Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]
        ],
        Dict[str, Union[torch.Tensor, np.ndarray]],
        Dataset,
        DataLoader,
    ],
    device: torch.device = torch.device("cpu"),
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Union[float, Dict[str, float]]:
    dataloader, _ = make_dataloaders(
        estimator=estimator,
        data=data,
        valid_percentage=0,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    estimator.eval()
    estimator = estimator.to(device)

    mis = collections.defaultdict(list)
    unfold = False
    for samples in dataloader:
        variables = unfold_samples(samples)
        variables = move_to_device(variables, device)

        estimation = estimator.mutual_information(**variables)

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
