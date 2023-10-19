from typing import Type, Optional, Dict, Any, Union, Tuple, List

import torch
import pandas as pd
from torch.optim import Optimizer
from tqdm.autonotebook import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    ConstantLR,
    LRScheduler,
)
from torch.utils.data import random_split, DataLoader

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.batch_utils import unfold_samples
from torch_mist.utils.data.dataset import SampleDataset
from torch_mist.utils.estimation import evaluate_mi


def _instantiate_dataloaders(
    x: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    train_loader: Optional[Any] = None,
    valid_loader: Optional[Any] = None,
    valid_percentage: float = 0.1,
    batch_size: Optional[int] = None,
    num_workers: int = 8,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if (x is None) != (y is None):
        raise ValueError(
            "Either both x and y need to be specified or neither."
        )

    if not ((x is None) ^ (train_loader is None)):
        raise ValueError(
            "Either both x and y or the train_loader need to be specified."
        )
    if not (x is None):
        if batch_size is None:
            raise ValueError("Please specify a value for batch_size.")

        # Make the dataloaders from the samples
        dataset = SampleDataset({"x": x, "y": y})
        if valid_percentage > 0:
            if not (valid_loader is None):
                raise ValueError(
                    "The valid_loader can't be specified when x and y are used"
                )

            n_valid = int(len(dataset) * valid_percentage)
            dataset, val_set = random_split(
                dataset, [len(dataset) - n_valid, n_valid]
            )
            valid_loader = DataLoader(
                val_set,
                batch_size=batch_size,
                num_workers=num_workers,
            )

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

    return train_loader, valid_loader


def _instantiate_optimizer(
    estimator: MIEstimator,
    max_epochs: int,
    iterations_per_epoch: int,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0.2,
) -> Tuple[Optimizer, LRScheduler]:
    opt_params = {"params": estimator.parameters()}

    if optimizer_params is None:
        optimizer_params = {"lr": 5e-4}

    opt_params.update(optimizer_params)

    opt = optimizer_class([opt_params])

    # Cosine annealing with initial linear warmup
    if lr_annealing:
        max_iterations = iterations_per_epoch * max_epochs
        if not 0 <= warmup_percentage <= 1:
            raise ValueError("Warmup percentage must be between 0 and 1")
        warmup_iterations = int(
            iterations_per_epoch * max_epochs * warmup_percentage
        )
        lr_scheduler = SequentialLR(
            opt,
            [
                LinearLR(
                    opt, start_factor=1e-2, total_iters=warmup_iterations
                ),
                CosineAnnealingLR(
                    opt,
                    T_max=max_iterations - warmup_iterations,
                    eta_min=1e-5,
                ),
            ],
            milestones=[warmup_iterations],
        )
    else:
        lr_scheduler = ConstantLR(opt, 1.0)

    return opt, lr_scheduler


def _train_epoch(
    estimator: MIEstimator,
    train_loader: DataLoader,
    opt: Optimizer,
    lr_scheduler: LRScheduler,
    epoch: int,
    iteration: int,
    device: Union[str, torch.device],
    return_log: bool,
    tqdm_iteration: Optional[tqdm] = None,
) -> List[Dict[str, Any]]:
    log = []
    trained_iterations = 0
    for samples in train_loader:
        x, y = unfold_samples(samples)

        x = x.to(device)
        y = y.to(device)

        loss = estimator.loss(x, y)

        if return_log:
            estimation = estimator(x, y)
            log.append(
                {
                    "loss": loss.item(),
                    "iteration": iteration + trained_iterations,
                    "value": estimation.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "split": "train",
                    "epoch": epoch + 1,
                }
            )

        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()
        trained_iterations += 1

        if tqdm_iteration:
            tqdm_iteration.update(1)
            tqdm_iteration.set_postfix_str(f"loss: {loss}")

    return log


def train_mi_estimator(
    estimator: MIEstimator,
    x: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    train_loader: Optional[Any] = None,
    valid_loader: Optional[Any] = None,
    valid_percentage: float = 0.1,
    batch_size: Optional[int] = None,
    num_workers: int = 8,
    device: Union[torch.device, str] = torch.device("cpu"),
    max_epochs: int = 10,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0.2,
    verbose: bool = True,
    return_log: bool = True,
    early_stopping: bool = True,
    patience: int = 3,
    delta: float = 0.001,
) -> Optional[pd.DataFrame]:
    train_loader, valid_loader = _instantiate_dataloaders(
        x=x,
        y=y,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_percentage=valid_percentage,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    opt, lr_scheduler = _instantiate_optimizer(
        estimator=estimator,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        lr_annealing=lr_annealing,
        warmup_percentage=warmup_percentage,
        max_epochs=max_epochs,
        iterations_per_epoch=len(train_loader),
    )

    estimator.train()
    estimator = estimator.to(device)
    log = []

    best_mi = 0
    iteration = 0
    tqdm_epochs = (
        tqdm(total=max_epochs, desc="Epoch", position=1) if verbose else None
    )
    tqdm_iteration = (
        tqdm(total=len(train_loader), desc="Iteration", position=1)
        if verbose
        else None
    )

    for epoch in range(max_epochs):
        if tqdm_epochs:
            tqdm_iteration.reset()

        train_log = _train_epoch(
            estimator=estimator,
            train_loader=train_loader,
            opt=opt,
            lr_scheduler=lr_scheduler,
            return_log=return_log,
            tqdm_iteration=tqdm_iteration,
            device=device,
            epoch=epoch,
            iteration=iteration,
        )
        log += train_log
        iteration += len(train_log)

        if valid_loader is not None:
            valid_mi = evaluate_mi(
                estimator=estimator, dataloader=valid_loader, device=device
            )
            if tqdm_epochs:
                tqdm_epochs.set_postfix_str(f"valid_mi: {valid_mi}")
            if return_log:
                log.append(
                    {
                        "value": valid_mi,
                        "epoch": epoch + 1,
                        "split": "validation",
                        "iteration": iteration,
                    }
                )

            if early_stopping:
                if estimator.lower_bound:
                    improvement = valid_mi - best_mi
                elif estimator.upper_bound:
                    improvement = best_mi - valid_mi
                else:
                    improvement = delta

                if improvement >= delta:
                    # Improvement
                    best_mi = valid_mi
                else:
                    patience -= 1

        if patience < 0:
            if verbose:
                print("No improvements on validation, stopping.")
            break

        if tqdm_epochs:
            tqdm_epochs.update(1)

    if return_log:
        log = pd.DataFrame(log)
        return log
