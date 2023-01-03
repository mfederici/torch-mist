from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class SimCLRScheduler(LambdaLR):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, cosine: bool = True, linear: bool = False):
        super().__init__(
            optimizer=optimizer,
            lr_lambda=linear_warmup_decay(
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                cosine=cosine,
                linear=linear
            )
        )
