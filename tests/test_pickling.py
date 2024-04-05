import os
import pickle
import tempfile
from copy import deepcopy

import torch

from torch_mist import estimate_mi
from torch_mist.data.multivariate import JointMultivariateNormal
from torch_mist.estimators import JS, TransformedMIEstimator, js
from torch_mist.critic import JointCritic
from torch import nn

from torch_mist.utils import train_mi_estimator
from torch_mist.utils.logging import PandasLogger
from torch_mist.utils.logging.metrics import compute_mean_std

n_dim = 5


def test_pickle():
    # First we define a critic network that maps pairs of samples to a scalar
    critic = JointCritic(
        joint_net=nn.Sequential(
            nn.Linear(n_dim * 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )
    )

    # We also specify the number of Monte Carlo samples to use for the estimation of the normalization constant
    neg_samples = 16

    # Then we pass it to the Jensen-Shannon estimator
    mi_estimator = JS(critic=critic, neg_samples=neg_samples)
    mi_estimator = TransformedMIEstimator(
        base_estimator=mi_estimator, transforms={"x": nn.Linear(n_dim, n_dim)}
    )

    p_xy = JointMultivariateNormal(sigma=1, rho=0.9, n_dim=n_dim)
    samples = p_xy.sample([10000])

    train_mi_estimator(
        estimator=mi_estimator,
        train_data=samples,
        batch_size=64,
        max_epochs=1,
        train_logged_methods=[
            "base_estimator.loss",
            "base_estimator.mutual_information",
        ],
        early_stopping=False,
    )

    print(mi_estimator)

    filepath = os.path.join(tempfile.gettempdir(), "js.pyt")
    torch.save(mi_estimator, filepath)
    print("Model Saved")

    mi_estimator = torch.load(filepath)
    print("Model Loaded")
    print(mi_estimator)


def test_pickle_estimate():
    p_xy = JointMultivariateNormal(sigma=1, rho=0.9, n_dim=n_dim)
    samples = p_xy.sample([10000])
    filepath = os.path.join(tempfile.gettempdir(), "trained.pyt")

    eval_logged_methods = [
        ("log_ratio", compute_mean_std),
        ("batch_loss", compute_mean_std),
    ]

    # TODO: fix serialization issues
    eval_logged_methods += [
        # ("unnormalized_log_ratio", compute_mean_std),
        "mutual_information"
    ]

    train_logged_methods = [("batch_loss", compute_mean_std)]

    estimate_mi(
        estimator="smile",
        data=samples,
        batch_size=64,
        max_epochs=1,
        trained_model_save_path=filepath,
        train_logged_methods=train_logged_methods,
        eval_logged_methods=eval_logged_methods,
    )

    mi_estimator = torch.load(filepath)
    print("Model Loaded")
    print(mi_estimator)
    os.remove(filepath)


def test_deepcopy():
    estimator = js(x_dim=1, y_dim=1, hidden_dims=[32])

    x = torch.zeros(10, 1)
    y = torch.zeros(10, 1)

    value_1 = estimator(x, y)

    estimator2 = deepcopy(estimator)

    value_2 = estimator2(x, y)
    assert value_1 == value_2
