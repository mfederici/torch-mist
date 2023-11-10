import torch


def select_off_diagonal(x: torch.Tensor, K: int) -> torch.Tensor:
    N = x.shape[0]

    # Consider the first K off-diagonal elements
    idx = torch.arange(N * K).to(x.device).view(N, K).long()
    idx = (idx % K + torch.div(idx, K, rounding_mode="floor") + 1) % N
    return x[idx.T]
