from typing import Tuple, Type, Optional, Dict, Any

import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer
from tqdm.auto import tqdm
from torch.optim import Adam

from torch_mist.estimators.base import MutualInformationEstimator


def _unfold_samples(samples):
    if isinstance(samples, tuple):
        if not len(samples) == 2:
            raise Exception("Dataloaders that iterate over tuples must have 2 elements")
        x, y = samples
    elif isinstance(samples, dict):
        if not ('x' in samples) or not ('y' in samples):
            raise Exception("Dataloaders that iterate over dictionaries must have the keys 'x' and 'y'")
        x = samples['x']
        y = samples['y']
    else:
        raise NotImplementedError(
            "The dataloader must iterate over pairs or dictionaries containing 'x' and 'y'"
        )

    return x, y


def optimize_mi_estimator(
        estimator: MutualInformationEstimator,
        dataloader: Any,
        device: torch.device = torch.device('cpu'),
        epochs: int = 1,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        return_log: bool = True,
) -> Optional[pd.DataFrame]:

    opt_params = {"params": estimator.parameters()}

    if optimizer_params is None:
        optimizer_params = {"lr": 5e-4}

    opt_params.update(optimizer_params)
    
    opt = optimizer_class(
        [opt_params]
    )

    log = []

    estimator.train()
    estimator = estimator.to(device)

    dl = tqdm(dataloader) if verbose else dataloader
    for epoch in range(epochs):
        for samples in dl:
            x, y = _unfold_samples(samples)

            x = x.to(device)
            y = y.to(device)

            loss = estimator.loss(x, y)

            if return_log:
                estimation = estimator(x, y)
                log.append({
                    'loss': loss.item(),
                    'iteration': len(log),
                    'value': estimation.item(),
                })

            opt.zero_grad()
            loss.backward()
            opt.step()

    if return_log:
        return pd.DataFrame(log)


def estimate_mi(
        estimator: MutualInformationEstimator,
        dataloader: Any,
        device: torch.device = torch.device('cpu'),
) -> Tuple[float, float]:
    mis = []

    estimator.eval()
    estimator = estimator.to(device)

    for samples in dataloader:
        x, y = _unfold_samples(samples)

        x = x.to(device)
        y = y.to(device)

        estimation = estimator(x, y)
        mis.append(estimation.item())

    return np.mean(mis), np.std(mis)
