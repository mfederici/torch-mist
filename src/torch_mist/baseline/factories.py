from typing import List, Any, Callable

from torch import nn

from torch_mist.baseline.base import LearnableBaseline, LearnableJointBaseline


def baseline_nn(
    x_dim: int, hidden_dims: List[int], nonlinearity: Callable = nn.ReLU(True)
) -> LearnableBaseline:
    from pyro.nn import DenseNN

    net = DenseNN(
        input_dim=x_dim,
        hidden_dims=hidden_dims,
        param_dims=[1],
        nonlinearity=nonlinearity,
    )

    return LearnableBaseline(net)


def joint_baseline_nn(
    x_dim: int,
    y_dim: int,
    hidden_dims: List[int],
    nonlinearity: Any = nn.ReLU(True),
) -> LearnableJointBaseline:
    from pyro.nn import DenseNN

    net = DenseNN(
        input_dim=x_dim + y_dim,
        hidden_dims=hidden_dims,
        param_dims=[1],
        nonlinearity=nonlinearity,
    )
    return LearnableJointBaseline(net)
