import collections
from typing import Optional, Union, Dict

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from torch_mist.utils.data.utils import prepare_variables, TensorDictLike


def evaluate(
    model: nn.Module,
    method: str,
    data: TensorDictLike,
    device: torch.device = torch.device("cpu"),
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Union[float, Dict[str, float]]:
    if not hasattr(model, method):
        raise ValueError(f"{model.__class__.__name__} has no method {method}.")

    # Make a dataloader if necessary
    if isinstance(data, DataLoader):
        dataloader = data
    else:
        if batch_size is None:
            raise ValueError("Plase specify a value for batch_size.")
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    model.eval()
    model = model.to(device)

    values = None
    for samples in dataloader:
        v_args, v_kwargs = prepare_variables(samples, device)
        estimation = getattr(model, method)(*v_args, **v_kwargs)

        if values is None:
            if isinstance(estimation, dict):
                values = collections.defaultdict(list)
            else:
                values = []

        if isinstance(estimation, dict):
            for key, value in estimation.items():
                values[key].append(value.item())
        else:
            values.append(estimation.item())

    if isinstance(values, dict):
        for k, v in values.items():
            values[k] = np.mean(v)
    else:
        values = np.mean(values)

    return values
