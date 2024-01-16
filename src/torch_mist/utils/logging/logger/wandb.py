import os
import tempfile
from typing import Dict, Any, Optional
import torch
from torch import nn

import wandb
from torch_mist.utils.logging.logger.base import Logger


class WandbLogger(Logger):
    def __init__(
        self, project: str, log_dir: Optional[str] = None, log_every: int = 10
    ):
        if log_dir is None:
            log_dir = tempfile.gettempdir()
        super().__init__(log_dir=log_dir, log_every=log_every)
        self.run = wandb.init(project=project)
        self.run.define_metric("epoch", hidden=True)
        self.run.define_metric("iteration", hidden=True)
        self.run.define_metric("train/*", step_metric="iteration")
        self.run.define_metric("valid/*", step_metric="epoch")
        self.project = project

    def add_config(self, config: Dict[str, Any]):
        self.run.config.update(config)

    def _log(
        self, data: Any, name: str, iteration: int, epoch: int, split: str
    ):
        name = name.replace(".", "/")
        if isinstance(data, dict):
            entry = {
                f"{name}/{key.replace('.','/')}": value
                for key, value in data.items()
            }
        else:
            entry = {name: data}

        entry = {f"{split}/{name}": value for name, value in entry.items()}

        entry["epoch"] = epoch
        entry["iteration"] = iteration

        wandb.log(entry, step=iteration)

    def get_log(self) -> wandb.wandb_sdk.wandb_run.Run:
        return self.run

    def _reset_log(self):
        self.run = wandb.init(project=self.project)

    def save_model(
        self, model: nn.Module, name: str, artifact_name: Optional[str] = None
    ):
        filepath = os.path.join(self.log_dir, name)
        torch.save(model, filepath)

        if artifact_name is None:
            artifact_name = name
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(filepath)
        self.run.log_artifact(artifact)

    def save_log(self):
        pass
