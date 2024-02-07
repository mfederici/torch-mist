import os.path
from typing import Dict, Any, Optional

import pandas as pd

from torch_mist.utils.logging.logger.base import Logger


class PandasLogger(Logger):
    def __init__(
        self,
        log_dir: str = ".",
        log_name: Optional[str] = None,
        log_every: int = 10,
    ):
        super().__init__(log_dir=log_dir, log_every=log_every)
        self.__log = []
        if log_name is None:
            log_name = "log.csv"
        if not log_name.endswith(".csv"):
            log_name = log_name + ".csv"
        self.log_name = log_name

    def _log(
        self, data: Any, name: str, iteration: int, epoch: int, split: str
    ):
        if not isinstance(data, Dict):
            data = {
                "mean": data,
            }
        data["name"] = name
        data["split"] = split
        data["iteration"] = iteration
        data["epoch"] = epoch
        self.__log.append(data)

    def get_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.__log)

    def _reset_log(self):
        self.__log = []

    def save_log(self):
        log = self.get_log()
        log_path = os.path.join(self.log_dir, self.log_name)
        log.to_csv(log_path)
