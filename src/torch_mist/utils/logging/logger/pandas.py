from typing import Dict, Any

import pandas as pd

from torch_mist.utils.logging.logger.base import Logger


class PandasLogger(Logger):
    def __init__(self):
        super().__init__()
        self.__log = []

    def _log(self, data: Dict[str, Any]):
        self.__log.append(data)

    def get_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.__log)

    def _reset(self):
        self.__log = []
