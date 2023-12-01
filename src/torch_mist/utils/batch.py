from typing import Dict, Tuple, Union
import torch


def unfold_samples(
    samples: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Unify the format for the samples
    """
    variables = {}
    if isinstance(samples, tuple) or isinstance(samples, list):
        if not len(samples) == 2:
            raise Exception(
                "Dataloaders that iterate over tuples must have 2 elements, use dictionaries for more larger tuples"
            )
        x, y = samples
        variables["x"] = x
        variables["y"] = y
    elif isinstance(samples, dict):
        variables = samples
    else:
        raise NotImplementedError(
            "The dataloader must iterate over pairs or dictionaries"
        )

    if len(variables) < 2:
        raise Exception(
            f"Not enough variables to unfold, at least 2 are required."
        )

    return variables


def move_to_device(
    variables: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    for name in variables:
        variables[name] = variables[name].to(device)
    return variables
