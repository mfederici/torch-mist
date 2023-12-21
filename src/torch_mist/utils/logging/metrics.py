from typing import Dict, Union

import torch


def compute_mean_std(input, output) -> Dict[str, float]:
    assert isinstance(output, torch.Tensor) or isinstance(output, dict)
    if isinstance(output, dict):
        summary = {}
        for name, value in output.items():
            name = str(name)
            assert isinstance(value, torch.Tensor)
            summary[f"{name}/mean"] = torch.mean(value).item()
            summary[f"{name}/std"] = torch.std(value).item()
        return summary
    else:
        return {
            "mean": torch.mean(output).item(),
            "std": torch.std(output).item(),
        }


def compute_mean(input, output) -> Union[float, Dict[str, float]]:
    assert isinstance(output, torch.Tensor) or isinstance(output, dict)
    if isinstance(output, dict):
        summary = {}
        for name, value in output.items():
            name = str(name)
            assert isinstance(value, torch.Tensor)
            summary[name] = value.mean().item()
        return summary
    else:
        return output.mean().item()
