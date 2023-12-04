import torch


def compute_mean_std(value):
    return [
        {"metric": "mean", "value": torch.mean(value).item()},
        {"metric": "std", "value": torch.std(value).item()},
    ]
