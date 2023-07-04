from typing import Iterator, Tuple, Dict, Type, Any

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
from torch.optim import Optimizer, Adam

from torch_mist.estimators import *
from torch_mist.quantization import FixedQuantization, trained_vector_quantization

from torch_mist.utils.data import SampleDataLoader
from torch_mist.utils.estimate import optimize_mi_estimator, estimate_mi

from torch_mist.distributions.joint import JointDistribution

x_dim = y_dim = 1
batch_size = 64
n_bins = 32
mc_samples = 16

optimizer_params = {"lr": 1e-3}
optimizer_class = Adam
n_train_samples = 100000
n_test_samples = 10000
n_pretrain_epochs = 3
hidden_dims = [64]
z_dim = 4
atol = 1e-1


def _make_data():
    rho = 0.9
    cov = torch.tensor([
        [1, rho],
        [rho, 1.]
    ])
    mean = torch.tensor([0., 0.])
    p_xy = MultivariateNormal(mean.unsqueeze(0), cov.unsqueeze(0))
    true_mi = (MultivariateNormal(mean, torch.eye(2)).entropy() - p_xy.entropy())
    entropy_y = Normal(0, 1).entropy()

    p_xy = JointDistribution(p_xy, dims=[1, 1], labels=['x', 'y'])
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

    return trainloader, testloader, true_mi, entropy_y


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
        mi_estimate, mi_std = estimate_mi(
            estimator,
            dataloader=testloader,
        )

        print("I(x;y)", mi_estimate, '+-', mi_std)

        # Check that the estimate is close to the true value
        assert np.isclose(mi_estimate, true_mi, atol=atol)


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
            mc_samples=mc_samples
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
        trained_vector_quantization(
            dataloader=trainloader,
            x_dim=x_dim,
            hidden_dims=hidden_dims+[z_dim],
            n_bins=n_bins,
            n_train_epochs=n_pretrain_epochs,
        ),
        trained_vector_quantization(
            dataloader=trainloader,
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims + [z_dim],
            n_bins=n_bins,
            n_train_epochs=n_pretrain_epochs,
            cross_modal=True,
            decoder_transform_params={"scale": 0.1}
        ),

    ]

    estimators = [
        pq(x_dim=x_dim, hidden_dims=hidden_dims, quantization=quantization) for quantization in quantizations
    ]
    estimators += [
        discrete(quantization_x=quantization, quantization_y=quantization) for quantization in quantizations
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


# def test_all():
#     # test_discriminative_estimators()
#     test_generative_estimators()
#     test_quantized_mi_estimators()