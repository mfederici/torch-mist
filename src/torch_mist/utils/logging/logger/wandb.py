import os
import tempfile
from typing import Dict, Any
import torch


from torch_mist.utils.logging.logger.base import Logger
import wandb


class WandbLogger(Logger):
    def __init__(self, project: str):
        super().__init__()
        self.run = wandb.init(project=project)
        self.project = project

    def _log(self, data: Any, name: str, context: Dict[str, Any]):
        name.replace(".", "/")
        if isinstance(data, dict):
            entry = {f"{name}/{key}": value for key, value in data.items()}
        else:
            entry = {name: data}

        if "split" in context:
            entry = {
                f'{context["split"]}/{name}': value
                for name, value in entry.items()
            }

        wandb.log(
            entry,
            step=context["iteration"] if "iteration" in context else None,
        )

    def get_log(self) -> wandb.wandb_sdk.wandb_run.Run:
        return self.run

    def _reset_log(self):
        self.run = wandb.init(project=self.project)

    def save(self, model, name):
        filepath = os.path.join(tempfile.gettempdir(), name)
        torch.save(model, filepath)

        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(filepath)
        self.run.log_artifact(artifact)
