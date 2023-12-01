from typing import List, Any, Callable

from torch import nn

from torch_mist.baseline.base import LearnableBaseline


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
