import os
import pickle
import tempfile
import torch
from torch_mist.estimators import JS
from torch_mist.critic import JointCritic
from torch import nn


def test_pickle():
    # First we define a critic network that maps pairs of samples to a scalar
    critic = JointCritic(
        joint_net=nn.Sequential(
            nn.Linear(10, 64),
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

    mi_estimator.loss(x=torch.randn(10, 5), y=torch.randn(10, 5))

    print(mi_estimator)

    pickle.dumps(mi_estimator)
    filepath = os.path.join(tempfile.gettempdir(), "js.pickle")

    with open(filepath, "wb") as f:
        pickle.dump(mi_estimator, f)
    print("Pickle dumps!")

    with open(filepath, "rb") as f:
        pickle.load(f)
    print("Pickle loads!")
