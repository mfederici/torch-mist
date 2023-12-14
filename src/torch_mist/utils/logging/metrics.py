from typing import Dict

import torch


def compute_mean_std(input, output) -> Dict[str, float]:
    return {"mean": torch.mean(output).item(), "std": torch.std(output).item()}


def compute_mean(input, output) -> float:
    assert isinstance(output, torch.Tensor)
    return output.mean().item()
