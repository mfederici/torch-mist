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
    LRScheduler,
)
from torch import nn
from torch.utils.data import DataLoader

from torch_mist.nn import Model
from torch_mist.utils.data.utils import (
    prepare_variables,
    TensorDictLike,
    make_default_dataloaders,
)

from torch_mist.utils.evaluation import evaluate
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.logger.base import Logger, DummyLogger
from torch_mist.utils.train.utils import RunTerminationManager


def instantiate_optimizer(
    model: nn.Module,
    max_iterations: int,
    warmup_iterations: int = 0,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    lr_annealing: bool = False,
) -> Tuple[Optimizer, LRScheduler]:
    params = [
        {"params": params}
        for params in model.parameters()
        if params.requires_grad
    ]

    if optimizer_params is None:
        optimizer_params = {"lr": 5e-4}

    opt = optimizer_class(params, **optimizer_params)

    # Cosine annealing with initial linear warmup
    if lr_annealing:
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
        lr_scheduler = None

    return opt, lr_scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    opt: Optimizer,
    device: Union[str, torch.device],
    lr_scheduler: Optional[LRScheduler] = None,
    train_method: str = "loss",
    eval_method: Optional[str] = None,
    logger: Optional[Logger] = None,
    tqdm_iteration: Optional[tqdm] = None,
    train_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    max_iterations: Optional[int] = None,
):
    if not hasattr(model, train_method):
        raise ValueError(
            f"{model.__class__.__name__} does not have a {train_method}() method."
        )

    if eval_method:
        if not hasattr(model, eval_method):
            raise ValueError(
                f"{model.__class__.__name__} does not have a {eval_method}() method."
            )

    with logger.train():
        model.train()
        with logger.epoch():
            for samples in train_loader:
                v_args, v_kwargs = prepare_variables(samples, device)

                if max_iterations:
                    if logger._iteration >= max_iterations:
                        break

                with logger.iteration():
                    with logger.logged_methods(model, train_logged_methods):
                        loss = getattr(model, train_method)(
                            *v_args, **v_kwargs
                        )

                    # Compute the evaluation only if necessary
                    if not (eval_method is None) and not isinstance(
                        logger, DummyLogger
                    ):
                        with logger.logged_methods(model, eval_logged_methods):
                            getattr(model, eval_method)(*v_args, **v_kwargs)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if not (lr_scheduler is None):
                        lr_scheduler.step()

                if tqdm_iteration:
                    tqdm_iteration.update(1)
                    tqdm_iteration.set_postfix_str(f"loss: {loss}")


def validate(
    model: nn.Module,
    eval_method: str,
    valid_loader: Optional[DataLoader],
    device: Union[str, torch.device],
    logger: Logger,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
) -> float:
    if eval_logged_methods is None:
        eval_logged_methods = []

    if valid_loader is not None:
        with logger.valid():
            with logger.logged_methods(
                model,
                eval_logged_methods,
            ):
                valid_score = evaluate(
                    model=model,
                    method=eval_method,
                    data=valid_loader,
                    device=device,
                )

            if isinstance(valid_score, dict):
                valid_score = sum(valid_score.values())
    else:
        valid_score = None

    return valid_score


def is_early_stopping_possible(
    valid_loader_missing: bool, eval_method_missing: bool, bound: bool
) -> bool:
    if valid_loader_missing:
        print(
            "[Warning]: Please specify a validation set or use valid_percentage>0 to use early_stopping."
        )
        return False
    if eval_method_missing:
        print("[Warning]: Please specify eval_method to use early_stopping.")
        return False
    if not bound:
        print(
            "[Warning]: early_stopping can be used only when maximize=True or minimize=True."
        )
        return False
    return True


def compute_training_time(
    iterations_per_epoch: int,
    max_epochs: int,
    max_iterations: int,
    warmup_percentage: float,
) -> Tuple[int, int, int]:
    if max_iterations is None and max_epochs is None:
        raise ValueError("Please specify either max_epochs or max_iterations")

    if max_epochs is None:
        max_epochs = int(np.ceil(max_iterations / iterations_per_epoch))

    if max_iterations is None:
        max_iterations = iterations_per_epoch * max_epochs

    if not 0 <= warmup_percentage <= 1:
        raise ValueError("Warmup percentage must be between 0 and 1")

    warmup_iterations = int(
        iterations_per_epoch * max_epochs * warmup_percentage
    )

    return max_epochs, max_iterations, warmup_iterations


def train_model(
    model: Model,
    train_data: TensorDictLike,
    train_method: str = "loss",
    eval_method: Optional[str] = None,
    valid_data: Optional[TensorDictLike] = None,
    valid_percentage: float = 0.1,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    device: Union[torch.device, str] = torch.device("cpu"),
    max_epochs: Optional[int] = None,
    max_iterations: Optional[int] = None,
    optimizer_class: Type[Optimizer] = Adam,
    optimizer_params: Optional[Dict[str, Any]] = None,
    lr_annealing: bool = False,
    warmup_percentage: float = 0,
    verbose: bool = True,
    logger: Optional[Union[Logger, bool]] = None,
    early_stopping: bool = False,
    patience: Optional[int] = None,
    tolerance: float = 0.001,
    fast_train: bool = False,
    train_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
    eval_logged_methods: Optional[
        List[Union[str, Tuple[str, Callable]]]
    ] = None,
) -> Optional[Any]:
    # Create the training and validation dataloaders
    train_loader, valid_loader = make_default_dataloaders(
        data=train_data,
        valid_data=valid_data,
        valid_percentage=valid_percentage,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Check if early stopping is possible
    if early_stopping:
        early_stopping = is_early_stopping_possible(
            valid_loader_missing=valid_loader is None,
            eval_method_missing=eval_method is None,
            bound=model.lower_bound or model.upper_bound,
        )

    # Determine the training duration
    max_epochs, max_iterations, warmup_iterations = compute_training_time(
        iterations_per_epoch=len(train_loader),
        max_epochs=max_epochs,
        max_iterations=max_iterations,
        warmup_percentage=warmup_percentage,
    )

    if patience is None and early_stopping:
        patience = int(max_epochs * 0.02)
        if patience < 1:
            patience = 1

        print(
            f"[Info]: patience is not specified, using patience={patience} (~2% of training epochs) by default."
        )

    # Instantiate the optimizer and lr_scheduler
    opt, lr_scheduler = instantiate_optimizer(
        model=model,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        lr_annealing=lr_annealing,
        warmup_iterations=warmup_iterations,
        max_iterations=max_iterations,
    )
    model = model.to(device)

    # Instantiate the logger
    # If the logger is None, use the default PandasLogger,
    if logger is None:
        logger = PandasLogger()
    # If False, instantiate a DummyLogger, which does not store any quantity
    elif logger is False:
        logger = DummyLogger()

    # If nothing is specified, log the loss and evaluation method
    if train_logged_methods is None:
        train_logged_methods = [train_method]
    if eval_logged_methods is None and not (eval_method is None):
        eval_logged_methods = [eval_method]

    # Run manager responsible for termination
    run_manager = RunTerminationManager(
        early_stopping=early_stopping,
        tolerance=tolerance,
        patience=patience,
        warmup_iterations=warmup_iterations,
        max_iterations=max_iterations,
        maximize=model.lower_bound,
        minimize=model.upper_bound,
        verbose=verbose,
    )

    # Bars for training
    tqdm_epochs = (
        tqdm(total=max_epochs, desc="Epoch", position=1) if verbose else None
    )
    tqdm_iteration = (
        tqdm(total=len(train_loader), desc="Iteration", position=2)
        if verbose
        else None
    )

    for epoch in range(max_epochs):
        if tqdm_epochs:
            tqdm_iteration.reset()

        # Train one epoch
        train_epoch(
            model=model,
            train_loader=train_loader,
            opt=opt,
            lr_scheduler=lr_scheduler,
            logger=logger,
            tqdm_iteration=tqdm_iteration,
            device=device,
            eval_method=eval_method if not fast_train else None,
            train_logged_methods=train_logged_methods,
            eval_logged_methods=eval_logged_methods,
            max_iterations=max_iterations,
        )

        # Compute the validation score
        valid_score = validate(
            model=model,
            eval_method=eval_method if eval_method else train_method,
            valid_loader=valid_loader,
            device=device,
            logger=logger,
            eval_logged_methods=eval_logged_methods,
        )

        # Update the bars
        if tqdm_epochs:
            if valid_score:
                s = [
                    f"valid_{eval_method if eval_method else train_method}: {np.round(valid_score,3)}"
                ]
                if early_stopping:
                    s += [f"patience: {run_manager.current_patience}"]
                    s += [f"best_value: {np.round(run_manager.best_value,3)}"]
                tqdm_epochs.set_postfix_str(", ".join(s))
            tqdm_epochs.update(1)

        # Determine if the training is over
        if run_manager.should_stop(
            iteration=logger._iteration, score=valid_score, model=model
        ):
            break

    if early_stopping and run_manager.current_patience > 0:
        print(
            "[Warning]: The train procedure ended since max_epoch or max_iteration has been reached."
            + "Consider increasing the training time by specifying larger values of max_epochs or max_iterations."
        )

    # Obtain the training log
    log = logger.get_log()

    # Load the state dictionary for the best score.
    # This works only if early_stopping is enabled
    run_manager.load_best_weights(model)

    return log
