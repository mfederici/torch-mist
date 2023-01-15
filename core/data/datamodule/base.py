from pytorch_lightning import LightningDataModule
import numpy as np

class DataModuleWithAttributes(LightningDataModule):
    def update_train_attributes(self, attributes: np.ndarray) -> None:
        raise NotImplementedError()

    def update_val_attributes(self, attributes: np.ndarray) -> None:
        raise NotImplementedError()

    def update_test_attributes(self, attributes: np.ndarray) -> None:
        raise NotImplementedError()
