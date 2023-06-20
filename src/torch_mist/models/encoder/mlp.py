from typing import List

from pyro import nn as pnn
from torchvision.models import ResNet


class EncoderMLP(pnn.DenseNN):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(EncoderMLP, self).__init__(input_dim, hidden_dims, [output_dim])
