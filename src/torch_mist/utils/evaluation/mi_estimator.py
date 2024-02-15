from typing import Optional, Dict, Union

import torch
from torch.utils.data import DataLoader

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.evaluation.model import evaluate

from torch_mist.utils.data.utils import (
    TensorDictLike,
    update_dataloader,
    make_dataset,
    make_default_dataloaders,
)


def evaluate_mi(
    estimator: MIEstimator,
    data: TensorDictLike,
    device: torch.device = torch.device("cpu"),
    batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> Union[float, Dict[str, float]]:
    # Make a dataloader if necessary
    dataloader, _ = make_default_dataloaders(
        data=data,
        batch_size=batch_size,
        num_workers=num_workers,
        valid_percentage=0,
    )

    dataloader = update_dataloader(estimator, dataloader)

    values = evaluate(
        model=estimator,
        method="mutual_information",
        data=dataloader,
        device=device,
    )

    if isinstance(values, dict):
        values = {
            f"I({x_key};{y_key})": v for (x_key, y_key), v in values.items()
        }

    return values
