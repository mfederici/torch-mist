from typing import Union, List, Dict
import torch
import numpy as np


def temporal_offset_data(
    sequence: Union[np.ndarray, torch.Tensor],
    lagtimes: Union[List[int], np.ndarray, torch.Tensor],
) -> Dict[str, Union[np.array, torch.Tensor]]:
    lagtimes = np.sort(lagtimes)
    if lagtimes[0] < 0 or lagtimes[-1] > len(sequence):
        raise ValueError("Invalid lagtimes.")
    lagtimes = np.concatenate([np.zeros(1), lagtimes]).astype(np.int32)
    offset_data = {}
    for lagtime in lagtimes:
        lagtime = int(lagtime)
        if lagtime == lagtimes[-1]:
            offset_data[f"t_{lagtime}"] = sequence[lagtime:]
        else:
            offset_data[f"t_{lagtime}"] = sequence[
                lagtime : -lagtimes[-1] + lagtime
            ]
    return offset_data
