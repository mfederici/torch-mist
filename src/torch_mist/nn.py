import torch.nn as nn


class Normalize(nn.Module):
    def forward(self, x):
        return x / (x**2).sum(-1).unsqueeze(-1) ** 0.5
