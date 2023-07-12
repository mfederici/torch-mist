from typing import Dict, Tuple, Union
import torch

def unfold_samples(samples: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unfold samples into x and y
    """
    if isinstance(samples, tuple):
        if not len(samples) == 2:
            raise Exception("Dataloaders that iterate over tuples must have 2 elements")
        x, y = samples
    elif isinstance(samples, dict):
        if not ('x' in samples) or not ('y' in samples):
            raise Exception("Dataloaders that iterate over dictionaries must have the keys 'x' and 'y'")
        x = samples['x']
        y = samples['y']
    else:
        raise NotImplementedError(
            "The dataloader must iterate over pairs or dictionaries containing 'x' and 'y'"
        )

    return x, y
