from typing import Tuple, Type, Optional, Dict, Any

import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer
from torch_mist.estimators import MutualInformationEstimator
from collections import Iterator
from tqdm.auto import tqdm
from torch.optim import Adam


def optimize_mi_estimator(
        estimator: MutualInformationEstimator,
        dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        n_epochs: int = 1,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:

    opt_params = {"params": estimator.parameters()}
    opt_params.update(optimizer_params)
    
    opt = optimizer_class(
        [opt_params]
    )

    log = []

    for epoch in range(n_epochs):
        for iteration, (x, y) in enumerate(tqdm(dataloader)):
            estimation = estimator(x, y)

            loss = estimation.loss

            log.append(
                {
                    'value': estimation.value.item(),
                    'loss': estimation.loss.item(),
                    'iteration': iteration,
                }
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

    return pd.DataFrame(log)


def estimate_mi(
        estimator: MutualInformationEstimator,
        dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
) -> float:
    mis = []

    for x, y in dataloader:
        estimation = estimator(x, y)
        mis.append(estimation.value.item())

    return np.mean(mis)


def estimate_mi_std(
        estimator: MutualInformationEstimator,
        dataloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[float, float]:
    mis = []

    for x, y in dataloader:
        estimation = estimator(x, y)
        mis.append(estimation.value.item())

    return np.mean(mis), np.std(mis)
