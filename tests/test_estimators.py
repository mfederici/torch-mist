from copy import deepcopy
from typing import Tuple, Dict, Type, Any

import numpy as np
import torch
from torch.optim import Optimizer, Adam
from pyro.distributions.transforms import conditional_affine_coupling
from torch.utils.data import DataLoader

from torch_mist.data.multivariate import JointMultivariateNormal
from torch_mist.distributions import conditional_transformed_normal
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
    MultiMIEstimator,
    js,
    flip_estimator,
    doe,
    nwj,
    pq,
    hybrid_pq,
    gm,
)
from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid import (
    ResampledHybridMIEstimator,
    ReweighedHybridMIEstimator,
    PQHybridMIEstimator,
)
from torch_mist.quantization import (
    FixedQuantization,
    vqvae,
    kmeans_quantization,
)
from torch_mist.utils.data import SampleDataset
from torch_mist.utils.data.dataset import DistributionDataset

from torch_mist.utils.evaluation import evaluate_mi
from torch_mist.utils.train.mi_estimator import train_mi_estimator


rho = 0.9
x_dim = y_dim = 1
batch_size = 64
n_bins = 32
neg_samples = 15
max_epochs = 1

optimizer_params = {"lr": 1e-3}
optimizer_class = Adam
n_train_samples = 50000
n_test_samples = 10000
n_pretrain_epochs = 3
hidden_dims = [64]
k_dim = 64
quantization_dim = 4
atol = 0.1


def _make_data() -> (
    Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]
):
    p_xy = JointMultivariateNormal(sigma=1, rho=rho, n_dim=1)
    true_mi = p_xy.mutual_information().item()
    entropy_y = p_xy.entropy("y")

    train_samples = p_xy.sample(torch.Size([n_train_samples]))
    test_samples = p_xy.sample(torch.Size([n_test_samples]))

    return train_samples, test_samples, true_mi, entropy_y


def _test_estimator(
    estimator: MIEstimator,
    train_samples: Dict[str, torch.Tensor],
    test_samples: Dict[str, torch.Tensor],
    true_mi: torch.Tensor,
    optimizer_params: Dict[str, Any],
    optimizer_class: Type[Optimizer],
    atol: float,
):
    # Train the estimator
    train_mi_estimator(
        estimator=estimator,
        train_data=train_samples,
        optimizer_params=optimizer_params,
        optimizer_class=optimizer_class,
        max_epochs=max_epochs,
        lr_annealing=False,
        batch_size=batch_size,
        verbose=False,
        early_stopping=estimator.lower_bound or estimator.upper_bound,
        num_workers=8,
    )

    # Compute the estimate
    mi_estimate = evaluate_mi(
        estimator,
        data=test_samples,
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
        flip_estimator(
            instantiate_estimator(
                estimator_name="nwj",
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dims=hidden_dims,
                neg_samples=neg_samples,
            )
        ),
        instantiate_estimator(
            estimator_name="infonce",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims + [k_dim],
            projection_head="symmetric",
        ),
        instantiate_estimator(
            estimator_name="infonce",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims + [k_dim],
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
            k_dim=k_dim,
            alpha=0.4,
        ),
        instantiate_estimator(
            estimator_name="flo",
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dims=hidden_dims,
            critic_type="joint",
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


def test_dummy_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    train_samples, test_samples, true_mi, _ = _make_data()

    estimators = [
        instantiate_estimator(
            estimator_name="dummy_generative",
        ),
        instantiate_estimator(estimator_name="dummy_discriminative"),
    ]

    for estimator in estimators:
        print(estimator)
        # Compute the estimate
        mi_estimate = evaluate_mi(
            estimator,
            data=test_samples,
            batch_size=batch_size,
        )

        assert np.isclose(mi_estimate, 0), f"Dummy Estimate: {mi_estimate}!=0"


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
            input_dim=y_dim, thresholds=torch.linspace(-3, 3, n_bins - 1)
        ),
        vqvae(
            input_dim=y_dim,
            hidden_dims=hidden_dims,
            quantization_dim=quantization_dim,
            n_bins=n_bins,
            max_epochs=n_pretrain_epochs,
            batch_size=batch_size,
        ),
        kmeans_quantization(
            n_bins=n_bins,
        ),
    ]

    estimators = [
        instantiate_estimator(
            estimator_name="pq",
            x_dim=x_dim,
            hidden_dims=hidden_dims,
            quantize_y=quantization,
        )
        for quantization in quantizations
    ]
    estimators += [
        instantiate_estimator(
            estimator_name="binned",
            quantize_x=quantization,
            quantize_y=quantization,
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


def test_hybrid_estimators():
    # Seed everything
    np.random.seed(0)
    torch.manual_seed(0)

    train_samples, test_samples, true_mi, entropy_y = _make_data()
    q_Y_given_X = conditional_transformed_normal(
        input_dim=y_dim,
        context_dim=x_dim,
        hidden_dims=hidden_dims,
        scale=(1 - rho**2) ** 0.1 + 0.1,
    )

    conditional_generative_estimator = doe(
        x_dim=x_dim,
        y_dim=y_dim,
        q_Y_given_X=q_Y_given_X,
        marginal_transform_name="linear",
    )

    joint_generative_estimator = gm(
        x_dim=x_dim,
        y_dim=y_dim,
        marginal_transform_name="affine_autoregressive",
        joint_transform_name="linear",
        hidden_dims=hidden_dims,
    )

    discriminative_estimator = nwj(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dims=hidden_dims,
        neg_samples=neg_samples,
    )

    quantize_y = kmeans_quantization(n_bins=8)

    estimators = [
        ResampledHybridMIEstimator(
            generative_estimator=deepcopy(conditional_generative_estimator),
            discriminative_estimator=deepcopy(discriminative_estimator),
        ),
        ResampledHybridMIEstimator(
            generative_estimator=deepcopy(joint_generative_estimator),
            discriminative_estimator=deepcopy(discriminative_estimator),
        ),
        ReweighedHybridMIEstimator(
            generative_estimator=deepcopy(conditional_generative_estimator),
            discriminative_estimator=deepcopy(discriminative_estimator),
        ),
        hybrid_pq(
            quantize_y=quantize_y,
            x_dim=x_dim,
            hidden_dims=hidden_dims,
            temperature=1,
            discriminative_estimator=deepcopy(discriminative_estimator),
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

    samples = p_xy.sample([50000])

    train_mi_estimator(
        estimator,
        train_data=samples,
        max_epochs=5,
        verbose=True,
        batch_size=batch_size,
        num_workers=8,
        valid_percentage=0,
    )

    mi_estimate = evaluate_mi(estimator, data=samples, batch_size=batch_size)

    print("True I(x;y): ", true_mi)
    print("Estimated I(x;y): ", mi_estimate)

    assert np.isclose(
        mi_estimate, true_mi, atol=atol
    ), f"Estimate {mi_estimate} is not close to true value {true_mi}."


def test_multi_estimator():
    train_samples, test_samples, true_mi, entropy_y = _make_data()

    # Create a new de-correlated variable
    train_samples["z"] = torch.roll(train_samples["x"], 1, 0)
    test_samples["z"] = torch.roll(test_samples["x"], 1, 0)

    train_set = SampleDataset(train_samples)
    test_set = SampleDataset(test_samples)

    estimator = MultiMIEstimator(
        estimators={
            ("x", "y"): js(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dims=hidden_dims,
                neg_samples=neg_samples,
            ),
            ("x", "z"): js(
                x_dim=x_dim,
                y_dim=y_dim,
                hidden_dims=hidden_dims,
                neg_samples=neg_samples,
            ),
        }
    )

    train_mi_estimator(
        estimator,
        train_data=train_set,
        max_epochs=5,
        verbose=False,
        batch_size=batch_size,
        num_workers=8,
    )

    mi_estimate = evaluate_mi(estimator, data=test_set, batch_size=batch_size)

    assert np.isclose(
        mi_estimate["I(x;y)"], true_mi, atol=atol
    ), f"Estimate {mi_estimate} is not close to true value {true_mi}."

    assert np.isclose(
        mi_estimate["I(x;z)"], 0, atol=atol
    ), f"Estimate {mi_estimate} is not close to true value {0}."
