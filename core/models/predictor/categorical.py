from typing import Dict, List, Any

from pyro.nn import DenseNN

from core.distributions.conditional import ConditionalCategorical



class ConditionalCategoricalMLP(ConditionalCategorical):
    def __init__(
            self,
            y_dim: int,
            n_classes: int,
            hidden_dims: List[int],
            a_dim: int = 1,
    ):

        net = DenseNN(input_dim=y_dim, hidden_dims=hidden_dims, param_dims=[n_classes] * a_dim)
        super().__init__(net)
