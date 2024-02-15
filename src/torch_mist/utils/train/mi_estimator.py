from typing import Type, Optional, Dict, Any, Union, Tuple, List, Callable

import torch
from torch.optim import Optimizer
from torch.optim import Adam
from torch.utils.data import DataLoader

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.data.dataset import WrappedDataset

from torch_mist.utils.logging.logger.base import Logger
from torch_mist.utils.data.utils import (
    TensorDictLike,
    make_default_dataloaders,
    update_dataloader,
)
from torch_mist.utils.train.model import train_model


def pretrain_components(
    estimator: MIEstimator, train_loader: DataLoader, verbose: bool = True
):
    for func, component in estimator._components_to_pretrain:
        trained = False
        if hasattr(component, "trained"):
            trained = component.trained
        if not trained:
            wrapped_loader = DataLoader(
                WrappedDataset(train_loader.dataset, func),
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
            )
            if verbose:
                print(f"Training {component.__class__.__name__}()")
            component.fit(wrapped_loader)


def train_mi_estimator(
    estimator: MIEstimator,
    data: TensorDictLike,
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
    warmup_percentage: float = 0.2,
    verbose: bool = True,
    logger: Optional[Union[Logger, bool]] = None,
    early_stopping: bool = False,
    patience: int = 5,
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
        data=data,
        valid_data=valid_data,
        valid_percentage=valid_percentage,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Pretrain all the components (such as quantization schemes) that need training
    pretrain_components(
        estimator=estimator, train_loader=train_loader, verbose=verbose
    )

    # Update the dataloader if the estimator requires custom batches
    train_loader = update_dataloader(
        estimator,
        train_loader,
    )

    # And the validation loader (if any)
    if not (valid_loader is None):
        valid_loader = update_dataloader(estimator, valid_loader)

    # Train the model
    return train_model(
        model=estimator,
        train_data=train_loader,
        valid_data=valid_loader,
        valid_percentage=0,
        early_stopping=early_stopping,
        patience=patience,
        tolerance=tolerance,
        batch_size=batch_size,
        max_epochs=max_epochs,
        lr_annealing=lr_annealing,
        warmup_percentage=warmup_percentage,
        max_iterations=max_iterations,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        maximize=estimator.lower_bound,
        minimize=estimator.lower_bound,
        eval_method="mutual_information",
        fast_train=fast_train,
        num_workers=num_workers,
        device=device,
        verbose=verbose,
        logger=logger,
        train_logged_methods=train_logged_methods,
        eval_logged_methods=eval_logged_methods,
    )
