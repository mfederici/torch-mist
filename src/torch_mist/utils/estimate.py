from typing import Tuple, Type, Optional, Dict, Any

import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR

from torch_mist.estimators.base import MutualInformationEstimator
from torch_mist.utils.batch_utils import unfold_samples


def optimize_mi_estimator(
        estimator: MutualInformationEstimator,
        train_loader: Any,
        valid_loader: Optional[Any] = None,
        device: torch.device = torch.device('cpu'),
        max_epochs: int = 10,
        optimizer_class: Type[Optimizer] = Adam,
        optimizer_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        return_log: bool = True,
        lr_annealing: bool = False,
        warmup_percentage: float = 0.2,
) -> Optional[pd.DataFrame]:

    opt_params = {"params": estimator.parameters()}

    if optimizer_params is None:
        optimizer_params = {"lr": 5e-4}

    opt_params.update(optimizer_params)
    
    opt = optimizer_class(
        [opt_params]
    )

    # Cosine annealing with initial linear warmup
    if lr_annealing:
        max_iterations = len(train_loader) * max_epochs
        if not 0 <= warmup_percentage <= 1:
            raise ValueError("Warmup percentage must be between 0 and 1")
        warmup_iterations = int(len(train_loader) * max_epochs * warmup_percentage)
        lr_scheduler = SequentialLR(
            opt,
            [
                LinearLR(
                    opt,
                    start_factor=1e-2,
                    total_iters=warmup_iterations
                ),
                CosineAnnealingLR(
                    opt,
                    T_max=max_iterations - warmup_iterations,
                    eta_min=1e-5,
                )
            ],
            milestones=[warmup_iterations]
        )
    else:
        lr_scheduler = ConstantLR(opt, 1.0)

    log = []
    valid_log = []

    estimator.train()
    estimator = estimator.to(device)

    def train_iteration(samples):
        x, y = unfold_samples(samples)

        x = x.to(device)
        y = y.to(device)

        loss = estimator.loss(x, y)

        if return_log:
            estimation = estimator(x, y)
            log.append({
                'loss': loss.item(),
                'iteration': len(log),
                'value': estimation.item(),
                'lr': lr_scheduler.get_last_lr()[0],
                'type': 'train',
                'epoch': epoch + 1,
            })

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()

    for epoch in range(max_epochs):
        if verbose:
            print(f"Epoch {epoch + 1} / {max_epochs}")
            for samples in tqdm(train_loader):
                train_iteration(samples)
        else:
            for samples in train_loader:
                train_iteration(samples)

        if valid_loader is not None:
            mi, mi_std = estimate_mi(estimator, valid_loader, device)
            if verbose:
                print(f"Validation MI: {mi} +- {mi_std}")
            if return_log:
                valid_log.append({
                    'value': mi,
                    'std': mi_std,
                    'epoch': epoch + 1,
                    'type': 'validation',
                    'iteration': len(log),
                })

    if return_log:
        log = pd.DataFrame(log)
        if valid_loader is not None:
            valid_log = pd.DataFrame(valid_log)
            log = pd.concat([log, valid_log])
        return log


def estimate_mi(
        estimator: MutualInformationEstimator,
        dataloader: Any,
        device: torch.device = torch.device('cpu'),
) -> Tuple[float, float]:
    mis = []

    estimator.eval()
    estimator = estimator.to(device)

    for samples in dataloader:
        x, y = unfold_samples(samples)

        x = x.to(device)
        y = y.to(device)

        estimation = estimator(x, y)
        mis.append(estimation.item())

    return np.mean(mis), np.std(mis)
