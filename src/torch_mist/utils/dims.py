from typing import Union, Dict, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from torch_mist.utils.batch import unfold_samples
from torch_mist.utils.misc import make_default_dataloaders


def infer_dims(
    data: Union[
        Tuple[
            Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]
        ],
        Dict[str, Union[torch.Tensor, np.ndarray]],
        Dataset,
        DataLoader,
    ],
) -> Dict[str, int]:
    dataloader, _ = make_default_dataloaders(
        data=data,
        valid_data=None,
        batch_size=1,
        valid_percentage=0,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    variables = unfold_samples(batch)
    return {k: v.shape[-1] for k, v in variables.items()}
