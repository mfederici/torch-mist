import torch
from torch import nn
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


class Constant(nn.Module):
    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer('value', value)

    def forward(self, *args, **kwargs):
        return self.value

    def __repr__(self):
        return f'Constant({self.value})'


class Identity(nn.Module):
    def forward(self, x):
        return x


class SkipDenseNN(DenseNN):
    def forward(self, x):
        y = super().forward(x)
        if not isinstance(y, list) and not isinstance(y, tuple):
            y = [y]
            was_list = False
        else:
            was_list = True

        # Add the input to each output
        for y_ in y:
            y_ += x

        if not was_list:
            y = y[0]

        return y


class MergeOutputs(nn.Module):
    def __init__(self, *nets):
        super().__init__()
        for net in nets:
            assert isinstance(net, nn.Module)
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        return [net(x) for net in self.nets]

