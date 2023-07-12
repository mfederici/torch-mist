from typing import Iterator, Tuple, Dict, Type, Any, Union

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
from torch.optim import Optimizer, Adam

from torch_mist.estimators import *
from torch_mist.quantization import FixedQuantization, vqvae_quantization

from torch_mist.utils.data import DistributionDataLoader
from torch_mist.utils.estimate import optimize_mi_estimator, estimate_mi

from torch_mist.distributions.joint import JointDistribution

rho = 0.9
x_dim = y_dim = 1
batch_size = 64
n_bins = 32
neg_samples = 16
max_epochs = 1

optimizer_params = {"lr": 1e-3}
optimizer_class = Adam
n_train_samples = 100000
n_test_samples = 10000
n_pretrain_epochs = 3
hidden_dims = [64]
quantization_dim = 4
atol = 1e-1


def _make_data():
    mean = torch.tensor([0., 0.])
    cov = torch.tensor([
        [1, rho],
        [rho, 1.]
    ])
    p_xy = MultivariateNormal(mean, cov)
    true_mi = (MultivariateNormal(mean, torch.eye(2)).entropy() - p_xy.entropy())
    entropy_y = Normal(0, 1).entropy()

    p_xy = JointDistribution(p_xy, dims=[1, 1], labels=['x', 'y'])
    trainloader = DistributionDataLoader(
        p_xy,
        batch_size=batch_size,
        max_samples=n_train_samples
    )

    testloader = DistributionDataLoader(
        p_xy,
        batch_size=batch_size,
        max_samples=n_test_samples,
    )

    return trainloader, testloader, true_mi, entropy_y


def _test_estimator(
        estimator: MutualInformationEstimator,
        trainloader: Iterator[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]],
        testloader: Iterator[Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]],
        true_mi: float,
        optimizer_params: Dict[str, Any],
        optimizer_class: Type[Optimizer],
        atol: float = 1e-1,
):

        # Train the estimator
        optimize_mi_estimator(
            estimator=estimator,
            train_loader=trainloader,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            max_epochs=max_epochs,
            lr_annealing=False,
        )

        # Compute the estimate
        mi_estimate, mi_std = estimate_mi(
            estimator,
            dataloader=testloader,
        )

        print("I(x;y)", mi_estimate, '+-', mi_std)

        # Check that the estimate is close to the true value
        assert np.isclose(mi_estimate, true_mi, atol=atol), \
            f"Estimate {mi_estimate} is not close to true value {true_mi}."


def test_discriminative_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    trainloader, testloader, true_mi, _ = _make_data()

    estimators = [
        nwj(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples
        ),
        infonce(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_params={"projection_head": "asymmetric"}
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
            neg_samples=neg_samples
        ),
        mine(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples
        ),
        smile(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples
        ),
        tuba(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples
        ),
        alpha_tuba(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples
        ),
        flo(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples
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


def test_generative_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    trainloader, testloader, true_mi, entropy_y = _make_data()

    estimators = [
        ba(
            x_dim=x_dim,
            y_dim=y_dim,
            entropy_y=entropy_y,
            hidden_dims=hidden_dims,
            transform_name='conditional_linear',
            n_transforms=1
        ),
        doe(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            conditional_transform_name='conditional_linear',
            n_conditional_transforms=1,
            marginal_transform_name='linear',
            n_marginal_transforms=1,
        ),
        gm(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            joint_transform_name='spline_autoregressive',
            n_joint_transforms=2,
            marginal_transform_name='linear',
            n_marginal_transforms=1,
        ),
        l1out(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name='conditional_linear',
            n_transforms=1
        ),
        club(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name='conditional_linear',
            n_transforms=1
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
            atol=atol if not isinstance(estimator, CLUB) else 10,
        )


def test_quantized_mi_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    trainloader, testloader, true_mi, _ = _make_data()

    quantizations = [
        FixedQuantization(
            input_dim=x_dim,
            thresholds=torch.linspace(-3, 3, n_bins - 1)
        ),
        vqvae_quantization(
            dataloader=trainloader,
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            quantization_dim=quantization_dim,
            n_bins=n_bins,
            max_epochs=n_pretrain_epochs,
        ),
        vqvae_quantization(
            dataloader=trainloader,
            input_dim=x_dim,
            target_dim=y_dim,
            hidden_dims=hidden_dims,
            quantization_dim=quantization_dim,
            n_bins=n_bins,
            max_epochs=n_pretrain_epochs,
            cross_modal=True,
            decoder_transform_params={"scale": 0.1}
        ),
    ]

    estimators = [
        pq(x_dim=x_dim, hidden_dims=hidden_dims, Q_x=quantization) for quantization in quantizations
    ]
    estimators += [
        discrete(Q_x=quantization, Q_y=quantization) for quantization in quantizations
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
