import os
import pickle
import tempfile
import torch

from torch_mist.data.multivariate import JointMultivariateNormal
from torch_mist.estimators import JS, TransformedMIEstimator
from torch_mist.critic import JointCritic
from torch import nn

from torch_mist.utils import train_mi_estimator
from torch_mist.utils.logging import PandasLogger


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
        x=samples["x"],
        y=samples["y"],
        batch_size=64,
        max_epochs=1,
        train_logged_methods=[
            "base_estimator.loss",
            "base_estimator.mutual_information",
        ],
    )

    print(mi_estimator)

    pickle.dumps(mi_estimator)
    filepath = os.path.join(tempfile.gettempdir(), "js.pickle")

    with open(filepath, "wb") as f:
        pickle.dump(mi_estimator, f)
    print("Pickle dumps!")

    with open(filepath, "rb") as f:
        pickle.load(f)
    print("Pickle loads!")
