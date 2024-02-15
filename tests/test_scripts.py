import os

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from sklearn.datasets import load_iris
from hydra import initialize, compose

from scripts.compute_mi import parse
from torch_mist.data.multivariate import JointMultivariateNormal
from torch_mist.utils import evaluate_mi
from torch_mist.utils.data import CSVDataset


def test_csv_script():
    csv_path = "./iris.csv"

    # Write the iris dataset to csv
    dataset = load_iris(as_frame=True)["data"]
    dataset.rename(
        columns={
            "sepal length (cm)": "sepal_1",
            "sepal width (cm)": "sepal_2",
            "petal length (cm)": "petal_1",
            "petal width (cm)": "petal_2",
        },
        inplace=True,
    )
    dataset.to_csv(csv_path, index=False)
    model_path = "trained_model.pyt"

    with initialize(version_base=None, config_path="../scripts/config"):
        # overrides
        cfg = compose(
            config_name="config",
            overrides=[
                "data=csv",
                "mi_estimator=js",
                f"data.filepath={csv_path}",
                f"save_trained_model={model_path}",
                "estimation.x_key=sepal",
                "estimation.y_key=petal",
            ],
        )

        parse(cfg)

    estimator = torch.load(model_path)
    data = CSVDataset(csv_path)

    mi = evaluate_mi(
        estimator=estimator,
        data=data,
        batch_size=64,
    )

    print(mi)

    assert "I(sepal;petal)" in mi
    assert np.isclose(mi["I(sepal;petal)"], 0.36, rtol=0.05)

    # Clean up
    del dataset
    os.remove(csv_path)
    os.remove(model_path)


def test_multimixture_script():
    model_path = "trained_model.pyt"

    with initialize(version_base=None, config_path="../scripts/config"):
        # overrides
        cfg = compose(
            config_name="config",
            overrides=[
                "data=multimixture",
                "mi_estimator=js",
                f"save_trained_model={model_path}",
                f"estimation.max_epochs=1",
                "metadata.x_dim=1",
                "metadata.y_dim=1",
                "save_train_log=true",
            ],
        )

        parse(cfg)

    estimator = torch.load(model_path)
    p_xy = instantiate(cfg.distribution)
    true_mi = p_xy.mutual_information().item()
    test_samples = p_xy.sample(torch.Size([10000]))

    mi = evaluate_mi(
        estimator=estimator,
        data=test_samples,
        batch_size=64,
    )

    assert os.path.isfile("log.csv")
    loaded_train_log = pd.read_csv("log.csv")
    assert len(loaded_train_log) == 409

    print(mi, true_mi)
    assert np.isclose(mi, true_mi, 0.4)
    os.remove(model_path)
    os.remove("log.csv")
