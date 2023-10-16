from typing import Optional, Iterator, Dict, Any, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch_mist.utils.data.dataset import SampleDataset


def train_vqvae(
    model: nn.Module,
    max_epochs: int,
    dataloader: Optional[Iterator] = None,
    x: Optional[Union[torch.Tensor, np.array]] = None,
    optimizer_class=torch.optim.Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 8,
) -> nn.Module:
    if (x is None and dataloader is None) or (
        not (x is None) and not (dataloader is None)
    ):
        raise ValueError(
            "Either a set of samples or a dataloader need to be specified."
        )
    if not (x is None):
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")
        dataset = SampleDataset({"x": x})
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    if optimizer_params is None:
        optimizer_params = dict(lr=1e-3)

    opt = optimizer_class(model.parameters(), **optimizer_params)

    for epoch in range(max_epochs):
        for samples in dataloader:
            if isinstance(samples, dict):
                if "x" not in samples:
                    raise ValueError("Expected 'x' key in samples")
                x = samples["x"]
            elif isinstance(samples, torch.Tensor):
                x = samples
            else:
                raise ValueError(
                    f"Unknown sample type {type(samples)}, expected dict containing keys 'x' and 'y' or torch.Tensor"
                )

            opt.zero_grad()
            model.loss(x).backward()
            opt.step()

    return model
