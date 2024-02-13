import os
from collections import defaultdict
from typing import Optional, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        remove_nan_rows: bool = True,
        rename_columns: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__()

        if not os.path.isfile(filepath):
            raise ValueError(f"{os.path.abspath(filepath)} does not exist.")

        df = pd.read_csv(filepath, **kwargs)
        if remove_nan_rows:
            df.dropna(inplace=True)

        self.variables = []
        self.variable_cols = defaultdict(list)
        self.data = {}
        last_variable = None
        for column in df.columns:
            idx = int(column.split("_")[-1])
            variable = "_".join(column.split("_")[:-1])
            self.variable_cols[variable].append(column)
            if variable != last_variable:
                assert not (variable in self.variables)
                self.variables.append(variable)
                last_variable = variable

            if idx != len(self.variable_cols[variable]):
                raise ValueError(
                    "To avoid ambiguities, please make sure the columns in the csv file are sorted"
                    + " from <name>_1 to <name>_N"
                )

        print(f"The file '{filepath}' has columns:")

        for variable in self.variables:
            self.data[variable] = df[self.variable_cols[variable]].values
            print(f"  '{variable}' with shape {self.data[variable].shape}")

    def __len__(self):
        return self.data[self.variables[0]].shape[0]

    def __getitem__(self, index):
        return {k: v[index].astype(np.float32) for k, v in self.data.items()}
