from typing import Optional, Iterator, Dict, Any, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_mist.utils.data.dataset import SampleDataset
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.logger.base import Logger, DummyLogger


def train_model(
    model: nn.Module,
    max_epochs: int,
    dataloader: Optional[Iterator] = None,
    data: Optional[Union[torch.Tensor, np.array]] = None,
    optimizer_class=torch.optim.Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 8,
    logger: Optional[Union[bool, Logger]] = None,
) -> Optional[Any]:
    if (data is None and dataloader is None) or (
        not (data is None) and not (dataloader is None)
    ):
        raise ValueError(
            "Either a set of samples or a dataloader need to be specified."
        )
    if not (data is None):
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")
        dataset = SampleDataset({"x": data})
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    if optimizer_params is None:
        optimizer_params = dict(lr=1e-3)

    opt = optimizer_class(model.parameters(), **optimizer_params)
    # If the logger is specified, we use it adding loss and mutual_information logs if not already specified
    # if it is None, use the default PandasLogger,
    # if False, instantiate a DummyLogger, which does not store any quantity
    if logger is None:
        logger = PandasLogger()
    elif logger is False:
        logger = DummyLogger()

    with logger.train():
        with logger.epoch():
            for epoch in range(max_epochs):
                for samples in dataloader:
                    if isinstance(samples, dict):
                        if "x" not in samples:
                            raise ValueError("Expected 'x' key in samples")
                        data = samples["x"]
                    elif isinstance(samples, torch.Tensor):
                        data = samples
                    else:
                        raise ValueError(
                            f"Unknown sample type {type(samples)}, expected dict containing keys 'x' and 'y' or torch.Tensor"
                        )
                    with logger.iteration():
                        opt.zero_grad()
                        model.loss(data).backward()
                        opt.step()

    train_log = logger.get_log()
    return train_log
