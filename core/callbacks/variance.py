from typing import Any, Optional, Union, List
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class VarianceLogCallback(Callback):
    def __init__(self, key: Union[str, List[str]], log_on: Optional[Union[str, List[str]]] = None) -> None:
        super().__init__()
        if isinstance(key, str):
            value_keys = [key]
        self.keys = key

        if log_on is None:
            log_on = ["train", "val"]
        self.values = {step: {key: [] for key in self.keys} for step in log_on}

    def on_any_batch_and(self, outputs: STEP_OUTPUT, step: str) -> None:
        if step in self.values:
            for key in self.values[step]:
                if key in outputs:
                    self.values[step][key].append(outputs[key].item())

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.on_any_batch_and(outputs, "train")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.on_any_batch_and(outputs, "val")

    def on_any_epoch_end(self, pl_module: pl.LightningModule, step: str) -> None:
        for key, values in self.values[step].items():
            if len(values) > 0:
                variance = np.std(values) ** 2
                pl_module.log(f"{key}/variance/{step}_epoch", variance)
            self.values[step][key] = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_any_epoch_end(pl_module, "train")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_any_epoch_end(pl_module, "val")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_any_epoch_end(pl_module, "test")