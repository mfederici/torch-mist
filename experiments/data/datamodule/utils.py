
import torch
class SignMapping:
    def __call__(self, x: torch.Tensor) -> torch.LongTensor:
        exponent = (x > 0).long()
        base = torch.arange(x.shape[-1]).view(1, -1).to(x.device)
        return (exponent * 2 ** base).sum(-1)
