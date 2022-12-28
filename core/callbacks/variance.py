from typing import Any
import numpy as np

from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class VarianceCallback(Callback):
    def __init__(self):
        super().__init__()
        self.grads = []
        self.values = []


    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.grads.append(outputs["loss"].item())
        self.values.append(outputs["value"].item())

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        grad_variance = np.std(self.grads) ** 2
        value_variance = np.std(self.values) ** 2
        pl_module.log("I(x;y)/train/grad_variance", grad_variance)
        pl_module.log("I(x;y)/train/value_variance", value_variance)
        self.grads = []
        self.values = []
