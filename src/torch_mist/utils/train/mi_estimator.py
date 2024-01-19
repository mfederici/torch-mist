from typing import Type, Optional, Dict, Any, Union, Tuple, List, Callable

import numpy as np
import torch
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
from torch.utils.data import DataLoader

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.batch import unfold_samples, move_to_device

from torch_mist.utils.evaluation import evaluate_mi
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.logger.base import Logger, DummyLogger
from torch_mist.utils.misc import make_dataloaders
from torch_mist.utils.train.utils import RunTerminationManager


def _instantiate_optimizer(
    estimator: MIEstimator,
    max_epochs: int,
    iterations_per_epoch: int,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0.2,
) -> Tuple[Optimizer, LRScheduler]:
    params = [
        {"params": params}
        for params in estimator.parameters()
        if params.requires_grad
    ]

    if optimizer_params is None:
        optimizer_params = {"lr": 5e-4}

    opt = optimizer_class(params, **optimizer_params)

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


def train_epoch(
    estimator: MIEstimator,
    train_loader: DataLoader,
    opt: Optimizer,
    lr_scheduler: LRScheduler,
    device: Union[str, torch.device],
    logger: Optional[Logger] = None,
    tqdm_iteration: Optional[tqdm] = None,
    fast_train: bool = False,
    train_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    max_iterations: Optional[int] = None,
):
    with logger.train():
        estimator.train()
        with logger.epoch():
            for samples in train_loader:
                variables = unfold_samples(samples)
                variables = move_to_device(variables, device)

                if max_iterations:
                    if logger._iteration >= max_iterations:
                        break

                with logger.iteration():
                    with logger.logged_methods(
                        estimator, train_logged_methods
                    ):
                        loss = estimator(**variables)

                    # Compute the ratio only if necessary
                    if not fast_train and not isinstance(logger, DummyLogger):
                        with logger.logged_methods(
                            estimator, eval_logged_methods
                        ):
                            estimator.mutual_information(**variables)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    lr_scheduler.step()

                if tqdm_iteration:
                    tqdm_iteration.update(1)
                    tqdm_iteration.set_postfix_str(f"loss: {loss}")


def validate(
    estimator: MIEstimator,
    valid_loader: Optional[DataLoader],
    device: Union[str, torch.device],
    logger: Logger,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
) -> float:
    if valid_loader is not None:
        with logger.valid():
            with logger.logged_methods(
                estimator,
                eval_logged_methods,
            ):
                valid_mi = evaluate_mi(
                    estimator=estimator,
                    dataloader=valid_loader,
                    device=device,
                )

            if isinstance(valid_mi, dict):
                valid_mi = sum(valid_mi.values())
    else:
        valid_mi = None

    return valid_mi


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
    max_epochs: Optional[int] = None,
    max_iterations: Optional[int] = None,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0.2,
    verbose: bool = True,
    logger: Optional[Union[Logger, bool]] = None,
    early_stopping: bool = False,
    patience: int = 3,
    delta: float = 0.001,
    fast_train: bool = False,
    train_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
) -> Optional[Any]:
    # Create the training and validation dataloaders
    train_loader, valid_loader = make_dataloaders(
        estimator=estimator,
        x=x,
        y=y,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_percentage=valid_percentage,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if max_epochs is None:
        if max_iterations is None:
            raise ValueError(
                "Please specify either max_epochs or max_iterations"
            )
        max_epochs = int(np.ceil(max_iterations / len(train_loader)))

    opt, lr_scheduler = _instantiate_optimizer(
        estimator=estimator,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        lr_annealing=lr_annealing,
        warmup_percentage=warmup_percentage,
        max_epochs=max_epochs,
        iterations_per_epoch=len(train_loader),
    )

    estimator = estimator.to(device)

    default_logger = False
    # If the logger is None, use the default PandasLogger,
    if logger is None:
        logger = PandasLogger()
        default_logger = True
    # If False, instantiate a DummyLogger, which does not store any quantity
    elif logger is False:
        logger = DummyLogger()
        default_logger = True

    # If nothing is specified, log the loss and mutual information
    if train_logged_methods is None:
        train_logged_methods = ["loss"]
    if eval_logged_methods is None:
        eval_logged_methods = ["mutual_information"]

    run_manager = RunTerminationManager(
        early_stopping=early_stopping,
        delta=delta,
        patience=patience,
        verbose=verbose,
        max_iterations=max_iterations,
        maximize=estimator.lower_bound,
        minimize=estimator.upper_bound,
    )

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

        train_epoch(
            estimator=estimator,
            train_loader=train_loader,
            opt=opt,
            lr_scheduler=lr_scheduler,
            logger=logger,
            tqdm_iteration=tqdm_iteration,
            device=device,
            fast_train=fast_train,
            train_logged_methods=train_logged_methods,
            eval_logged_methods=eval_logged_methods,
            max_iterations=max_iterations,
        )

        valid_mi = validate(
            estimator=estimator,
            valid_loader=valid_loader,
            device=device,
            logger=logger,
            eval_logged_methods=eval_logged_methods,
        )

        if tqdm_epochs:
            if valid_mi:
                tqdm_epochs.set_postfix_str(f"valid_mi: {valid_mi}")
            tqdm_epochs.update(1)

        if run_manager.should_stop(
            iteration=logger._iteration, valid_mi=valid_mi
        ):
            break

    log = logger.get_log()

    if default_logger:
        logger.clear()

    return log
