from typing import Iterator, Tuple, Dict, Type, Any, Union

import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal
from torch.optim import Optimizer, Adam

from torch_mist.estimators import (
    MutualInformationEstimator,
    instantiate_estimator,
    CLUB,
)
from torch_mist.quantization import FixedQuantization, vqvae_quantization

from torch_mist.utils.estimation import evaluate_mi
from torch_mist.train.mi_estimator import train_mi_estimator

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
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1, rho], [rho, 1.0]])
    p_xy = MultivariateNormal(mean, cov)
    true_mi = MultivariateNormal(mean, torch.eye(2)).entropy() - p_xy.entropy()
    entropy_y = Normal(0, 1).entropy()

    p_xy = JointDistribution(p_xy, dims=[1, 1], labels=["x", "y"])
    train_samples = p_xy.sample([n_train_samples])
    test_samples = p_xy.sample([n_test_samples])

    return train_samples, test_samples, true_mi, entropy_y


def _test_estimator(
    estimator: MutualInformationEstimator,
    train_samples: Dict[str, torch.Tensor],
    test_samples: Dict[str, torch.Tensor],
    true_mi: float,
    optimizer_params: Dict[str, Any],
    optimizer_class: Type[Optimizer],
    atol: float = 1e-1,
):
    # Train the estimator
    train_mi_estimator(
        estimator=estimator,
        x=train_samples["x"],
        y=train_samples["y"],
        optimizer_params=optimizer_params,
        optimizer_class=optimizer_class,
        max_epochs=max_epochs,
        lr_annealing=False,
        batch_size=batch_size,
        verbose=False,
        return_log=True,
    )

    # Compute the estimate
    mi_estimate = evaluate_mi(
        estimator,
        x=test_samples["x"],
        y=test_samples["y"],
        batch_size=batch_size,
    )

    print("I(x;y): ", mi_estimate)

    # Check that the estimate is close to the true value
    assert np.isclose(
        mi_estimate, true_mi, atol=atol
    ), f"Estimate {mi_estimate} is not close to true value {true_mi}."


def test_discriminative_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    train_samples, test_samples, true_mi, _ = _make_data()

    estimators = [
        instantiate_estimator(
            estimator_name="nwj",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
        instantiate_estimator(
            estimator_name="nwj",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
            critic_type="separable",
        ),
        instantiate_estimator(
            estimator_name="infonce",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_params={"projection_head": "asymmetric"},
        ),
        instantiate_estimator(
            estimator_name="infonce",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
        ),
        instantiate_estimator(
            estimator_name="infonce",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
        instantiate_estimator(
            estimator_name="js",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
        instantiate_estimator(
            estimator_name="mine",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
        instantiate_estimator(
            estimator_name="smile",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
        instantiate_estimator(
            estimator_name="tuba",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
        instantiate_estimator(
            estimator_name="alpha_tuba",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
        instantiate_estimator(
            estimator_name="alpha_tuba",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
            learnable_baseline=False,
        ),
        instantiate_estimator(
            estimator_name="flo",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            neg_samples=neg_samples,
        ),
    ]

    for estimator in estimators:
        print(estimator)
        _test_estimator(
            estimator=estimator,
            train_samples=train_samples,
            test_samples=test_samples,
            true_mi=true_mi,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            atol=atol,
        )


def test_generative_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    train_samples, test_samples, true_mi, entropy_y = _make_data()

    estimators = [
        instantiate_estimator(
            estimator_name="ba",
            x_dim=x_dim,
            y_dim=y_dim,
            entropy_y=entropy_y,
            hidden_dims=hidden_dims,
            transform_name="conditional_linear",
            n_transforms=1,
        ),
        instantiate_estimator(
            estimator_name="doe",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            conditional_transform_name="conditional_linear",
            n_conditional_transforms=1,
            marginal_transform_name="linear",
            n_marginal_transforms=1,
        ),
        instantiate_estimator(
            estimator_name="gm",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            joint_transform_name="spline_autoregressive",
            n_joint_transforms=2,
            marginal_transform_name="linear",
            n_marginal_transforms=1,
        ),
        instantiate_estimator(
            estimator_name="l1out",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name="conditional_linear",
            n_transforms=1,
        ),
        instantiate_estimator(
            estimator_name="club",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            transform_name="conditional_linear",
            n_transforms=1,
        ),
    ]

    for estimator in estimators:
        print(estimator)
        _test_estimator(
            estimator=estimator,
            train_samples=train_samples,
            test_samples=test_samples,
            true_mi=true_mi,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            atol=atol if not isinstance(estimator, CLUB) else 10,
        )


def test_quantized_mi_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    train_samples, test_samples, true_mi, _ = _make_data()

    quantizations = [
        FixedQuantization(
            input_dim=x_dim, thresholds=torch.linspace(-3, 3, n_bins - 1)
        ),
        vqvae_quantization(
            x=train_samples["x"],
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            quantization_dim=quantization_dim,
            n_bins=n_bins,
            max_epochs=n_pretrain_epochs,
            batch_size=batch_size,
        ),
        vqvae_quantization(
            x=train_samples["x"],
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            quantization_dim=quantization_dim,
            n_bins=n_bins,
            decoder_transform_params={"scale": 0.1},
            batch_size=batch_size,
        ),
    ]

    estimators = [
        instantiate_estimator(
            estimator_name="pq",
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            Q_x=quantization,
        )
        for quantization in quantizations
    ]
    estimators += [
        instantiate_estimator(
            estimator_name="discrete", Q_x=quantization, Q_y=quantization
        )
        for quantization in quantizations
    ]

    for estimator in estimators:
        print(estimator)
        _test_estimator(
            estimator=estimator,
            train_samples=train_samples,
            test_samples=test_samples,
            true_mi=true_mi,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            atol=atol,
        )
