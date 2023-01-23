from typing import Type, Any
import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedDataset
from pytorch_lightning import LightningDataModule

from core.data.datamodule.base import DataModuleWithAttributes
from core.data.sampler.same_attributes import SameAttributesSampler
from core.task import InfoMax

torch.multiprocessing.set_sharing_strategy('file_system')

class ClusteringCallback(Callback):
    def __init__(
            self,
            steps: int,
            clustering: Any,
            key_to_encode: str = 'x',
    ):
        super().__init__()

        assert hasattr(clustering, 'fit'), "clustering must have a fit method"
        assert hasattr(clustering, 'predict'), "clustering must have a predict method"

        self.steps = steps
        self.clustering = clustering
        self.key_to_encode = key_to_encode
        self.clustered = False

    def _encode_all(self, encoder: nn.Module, dataloader: DataLoader) -> np.ndarray:
        encoder.eval()
        z = []
        ids = []
        device = next(encoder.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                assert self.key_to_encode in batch, f"key {self.key_to_encode} not in batch"
                x = batch[self.key_to_encode].to(device)
                z.append(encoder(x).cpu().numpy())
                ids.append(batch['idx'].cpu().numpy())
        # Concatenate the batches
        ids = np.concatenate(ids)
        z = np.concatenate(z)[np.argsort(ids)]

        # Return the encoded data
        encoder.train()
        return z

    def _update_sampling(self, a: np.ndarray, dataloader: DataLoader) -> None:
        a = a.reshape(-1, 1)
        assert isinstance(dataloader.batch_sampler, SameAttributesSampler), "ClusteringCallback only works with SameAttributesSampler"
        if dataloader.batch_sampler.attributes is None:
            dataloader.batch_sampler.attributes = a
        else:
            dataloader.batch_sampler.attributes = np.concatenate([dataloader.batch_sampler.attributes, a], dim=1)

    def _update_data(self, a: np.ndarray, dataloader: DataLoader) -> None:
        dataset = dataloader.dataset
        a = a.reshape(-1, 1)
        if isinstance(dataset, CombinedDataset):
            dataset = dataset.datasets
        assert hasattr(dataset, 'attributes'), "dataset must have an attribute called attributes"
        if dataset.attributes is None:
            dataset.attributes = a
        else:
            dataset.attributes = np.concatenate([dataset.attributes, a], dim=1)

    def on_train_batch_start(
            self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:

        assert isinstance(pl_module, InfoMax), "ClusteringCallback only works with InfoMax"
        if trainer.global_step >= self.steps and not self.clustered:
            self.clustered = True
            assert hasattr(trainer, 'datamodule'), "Trainer must have a datamodule"
            assert isinstance(trainer.datamodule, DataModuleWithAttributes), "ClusteringCallback only works with DataModuleWithAttributes"

            print("ClusteringCallback: Encoding all training data...")
            z_train = self._encode_all(pl_module.encoder_x, trainer.datamodule.train_dataloader())

            print("ClusteringCallback: Clustering...")
            self.clustering.fit(z_train)

            a_train = self.clustering.predict(z_train)

            # Compute the entropy of the attributes
            p_a = np.bincount(a_train) / len(a_train)
            h_a = -np.sum(p_a[p_a > 0] * np.log(p_a[p_a > 0]))

            print(f"Entropy: {h_a}")

            assert hasattr(trainer, 'datamodule'), "Trainer must have a datamodule"
            assert isinstance(trainer.datamodule, DataModuleWithAttributes), "ClusteringCallback only works with InfoMax"
            trainer.datamodule.update_train_attributes(a_train)

            if len(trainer.val_dataloaders) > 0:
                assert len(trainer.val_dataloaders) == 1, "ClusteringCallback only works with one validation dataloader"
                print("ClusteringCallback: Encoding all validation data...")
                z_val = self._encode_all(pl_module.encoder_x, trainer.datamodule.val_dataloader())

                a_val = self.clustering.predict(z_val)

                trainer.datamodule.update_val_attributes(a_val)
                # self._update_sampling(a_val, trainer.val_dataloaders[0])
                # self._update_data(a_val, trainer.val_dataloaders[0])

            print("ClusteringCallback: Done")



