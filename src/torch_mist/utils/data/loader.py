from typing import Dict, Union, Callable, Any

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from torch_mist.utils.data.sampler import SameAttributeSampler


class SameAttributeDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        attributes: Union[torch.Tensor, np.ndarray],
        batch_size: int,
        neg_samples: int,
        **kwargs
    ):
        if len(dataset) != len(attributes):
            raise Exception(
                "The dataset and the attributes must have the same length."
            )

        super().__init__(
            dataset,
            batch_sampler=SameAttributeSampler(
                attributes=attributes,
                batch_size=batch_size,
                neg_samples=neg_samples,
            ),
            **kwargs
        )


def sample_same_attributes(
    dataloader: DataLoader,
    attributes: Union[torch.Tensor, np.ndarray],
    neg_samples: int,
) -> DataLoader:
    return SameAttributeDataLoader(
        neg_samples=neg_samples,
        dataset=dataloader.dataset,
        attributes=attributes,
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        prefetch_factor=dataloader.prefetch_factor,
        pin_memory=dataloader.pin_memory,
        pin_memory_device=dataloader.pin_memory_device,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
    )


def sample_same_value(
    dataloader: DataLoader,
    func: Callable[[Any], Any],
    neg_samples: int,
) -> DataLoader:
    ordered_dataloader = DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        prefetch_factor=dataloader.prefetch_factor,
        pin_memory=dataloader.pin_memory,
        pin_memory_device=dataloader.pin_memory_device,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        shuffle=False,
    )

    attributes = []
    for batch in ordered_dataloader:
        attributes.append(func(batch))

    attributes = torch.cat(attributes, 0)

    return sample_same_attributes(
        dataloader, attributes, neg_samples=neg_samples
    )
