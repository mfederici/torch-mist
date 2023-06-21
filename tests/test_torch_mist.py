from typing import Iterator, Tuple, Dict, Type

import numpy as np
from torch.distributions import Distribution, MultivariateNormal
from torch.optim import Optimizer, Adam

from torch_mist.estimators.discriminative import *
from torch_mist.utils.data import SampleDataLoader
from torch_mist.utils.estimate import optimize_mi_estimator, estimate_mi


def _test_estimator(
        estimator: MutualInformationEstimator,
        trainloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        testloader: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        true_mi: float,
        optimizer_params: Dict[str, Any],
        optimizer_class: Type[Optimizer],
        atol: float = 1e-1,
):

        # Train the estimator
        optimize_mi_estimator(
            estimator=estimator,
            dataloader=trainloader,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
        )

        # Compute the estimate
        mi_estimate = estimate_mi(
            estimator,
            dataloader=testloader,
        )

        print("I(x;y)", mi_estimate)

        # Check that the estimate is close to the true value
        assert np.isclose(mi_estimate, true_mi, atol=atol)


def test_discriminative_estimators():
    batch_size = 64
    optimizer_params = {"lr": 1e-3}
    optimizer_class = Adam
    n_train_samples = 100000
    n_test_samples = 10000
    mc_samples = 16
    hidden_dims = [64]
    atol = 1e-1

    rho = 0.9
    cov = torch.tensor([
        [1, rho],
        [rho, 1.]
    ])
    mean = torch.tensor([0., 0.])
    p_xy = MultivariateNormal(mean, cov)
    x_dim = y_dim = 1
    true_mi = (MultivariateNormal(mean, torch.eye(2)).entropy() - p_xy.entropy())

    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    trainloader = SampleDataLoader(
        p_xy,
        batch_size=batch_size,
        max_samples=n_train_samples
    )

    testloader = SampleDataLoader(
        p_xy,
        batch_size=batch_size,
        max_samples=n_test_samples,
    )

    estimators = [
        nwj(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            mc_samples=mc_samples
        ),
        infonce(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        ),
        js(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            mc_samples=mc_samples
        ),
        mine(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            mc_samples=mc_samples
        ),
        smile(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            mc_samples=mc_samples
        ),
        tuba(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            mc_samples=mc_samples
        ),
        alpha_tuba(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            mc_samples=mc_samples
        ),
        flo(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            mc_samples=mc_samples
        ),
    ]

    for estimator in estimators:
        print(estimator)
        _test_estimator(
            estimator=estimator,
            trainloader=trainloader,
            testloader=testloader,
            true_mi=true_mi,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            atol=atol,
        )


