from typing import Tuple, Dict, Type, Any

import numpy as np
import torch
from torch.optim import Optimizer, Adam
from pyro.distributions.transforms import conditional_affine_coupling

from torch_mist.data.multivariate import JointMultivariateNormal
from torch_mist.distributions.normal import ConditionalStandardNormalModule
from torch_mist.distributions.transforms import (
    ConditionalTransformedDistributionModule,
    permute,
)
from torch_mist.estimators import (
    MIEstimator,
    instantiate_estimator,
    CLUB,
    BA,
)
from torch_mist.quantization import FixedQuantization, vqvae_quantization
from torch_mist.utils.data import DistributionDataLoader

from torch_mist.utils.evaluation import evaluate_mi
from torch_mist.utils.train.mi_estimator import train_mi_estimator


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
output_dim = 64
quantization_dim = 4
atol = 1e-1


def _make_data() -> (
    Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]
):
    p_xy = JointMultivariateNormal(sigma=1, rho=rho, n_dim=1)
    true_mi = p_xy.mutual_information()
    entropy_y = p_xy.entropy("y")

    train_samples = p_xy.sample([n_train_samples])
    test_samples = p_xy.sample([n_test_samples])

    return train_samples, test_samples, true_mi, entropy_y


def _test_estimator(
    estimator: MIEstimator,
    train_samples: Dict[str, torch.Tensor],
    test_samples: Dict[str, torch.Tensor],
    true_mi: torch.Tensor,
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
    )

    # Compute the estimate
    mi_estimate = evaluate_mi(
        estimator,
        x=test_samples["x"],
        y=test_samples["y"],
        batch_size=batch_size,
    )

    print("True I(x;y): ", true_mi)
    print("Estimated I(x;y): ", mi_estimate)

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
            estimator_name="infonce",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            projection_head="symmetric",
        ),
        instantiate_estimator(
            estimator_name="infonce",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
        ),
        instantiate_estimator(
            estimator_name="js",
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
            critic_type="separable",
            output_dim=output_dim,
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
            data=train_samples["x"],
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            quantization_dim=quantization_dim,
            n_bins=n_bins,
            max_epochs=n_pretrain_epochs,
            batch_size=batch_size,
        ),
        vqvae_quantization(
            data=train_samples["x"],
            input_dim=x_dim,
            hidden_dims=hidden_dims,
            quantization_dim=quantization_dim,
            n_bins=n_bins,
            batch_size=batch_size,
            beta=0.01,
        ),
    ]

    estimators = [
        instantiate_estimator(
            estimator_name="pq",
            x_dim=x_dim,
            hidden_dims=hidden_dims,
            Q_y=quantization,
        )
        for quantization in quantizations
    ]
    estimators += [
        instantiate_estimator(
            estimator_name="binned", Q_x=quantization, Q_y=quantization
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


def test_flow_generative():
    # Seed everything
    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 2

    p_xy = JointMultivariateNormal(n_dim=input_dim)
    true_mi = p_xy.mutual_information()
    entropy_y = p_xy.entropy("y")

    base = ConditionalStandardNormalModule(input_dim)
    transforms = [
        conditional_affine_coupling(
            input_dim=input_dim, context_dim=input_dim, hidden_dims=hidden_dims
        ),
        permute(input_dim),
        conditional_affine_coupling(
            input_dim=input_dim, context_dim=input_dim, hidden_dims=hidden_dims
        ),
        permute(input_dim),
        conditional_affine_coupling(
            input_dim=input_dim, context_dim=input_dim, hidden_dims=hidden_dims
        ),
    ]
    transformed_dist = ConditionalTransformedDistributionModule(
        base, transforms
    )

    estimator = BA(
        q_Y_given_X=transformed_dist,
        entropy_y=entropy_y,
    )

    train_loader = DistributionDataLoader(
        joint_dist=p_xy,
        batch_size=64,
        max_samples=100000,
    )

    train_mi_estimator(
        estimator,
        train_loader=train_loader,
        max_epochs=5,
        verbose=False,
    )

    mi_estimate = evaluate_mi(estimator, dataloader=train_loader)

    print("True I(x;y): ", true_mi)
    print("Estimated I(x;y): ", mi_estimate)

    assert np.isclose(
        mi_estimate, true_mi, atol=atol
    ), f"Estimate {mi_estimate} is not close to true value {true_mi}."
