import torch
from torch_mist.utils.caching import cached_function
from typing import Tuple


@cached_function
def _make_indexing_mask(n: int, k: int, device: torch.Tensor) -> torch.Tensor:
    # Consider the first K off-diagonal elements
    idx = torch.arange(n * k).to(device).view(n, k).long()
    idx = (idx % k + torch.div(idx, k, rounding_mode="floor") + 1) % n
    return idx.T


def select_k_others(x: torch.Tensor, k: int) -> torch.Tensor:
    N = x.shape[0]
    idx = _make_indexing_mask(N, k, x.device)
    return x[idx]


@cached_function
def _make_off_selection_indexing_mask(
    n: int, k: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    cols = torch.arange(n).unsqueeze(0).repeat(k, 1)
    rows = (cols + 1 + torch.arange(k).unsqueeze(1)) % n
    return rows.to(device), cols.to(device)


def matrix_off_diagonal(matrix: torch.Tensor, k: int):
    N = matrix.shape[0]
    rows, cols = _make_off_selection_indexing_mask(N, k, matrix.device)
    res = matrix[rows, cols, ...]
    return res
