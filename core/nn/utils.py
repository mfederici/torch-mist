import torch
from pyro.nn import DenseNN


class JointDenseNN(DenseNN):
    def __init__(self, input_dims, hidden_dims, param_dims, dim=-1):
        super(JointDenseNN, self).__init__(input_dim=sum(input_dims), hidden_dims=hidden_dims, param_dims=param_dims)
        self.dim = dim
        self.input_dims = input_dims
        
    def forward(self, *x):
        assert len(x) == len(self.input_dims)
        xy = torch.cat(x, self.dim)        
        return super(JointDenseNN, self).forward(xy)
