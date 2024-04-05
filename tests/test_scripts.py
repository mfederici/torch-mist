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
                f"estimation.trained_model_save_path={model_path}",
                "estimation.x_key=sepal",
                "estimation.y_key=petal",
            ],
        )

        parse(cfg)

    estimator = torch.load(model_path)
    data = CSVDataset(csv_path)

    # Clean up
    os.remove(csv_path)
    os.remove(model_path)

    mi = evaluate_mi(
        estimator=estimator,
        data=data,
        batch_size=64,
    )

    print(mi)

    assert "I(sepal;petal)" in mi
    assert np.isclose(mi["I(sepal;petal)"], 0.8, rtol=0.05)


def test_multimixture_script():
    model_path = "trained_model.pyt"
    total_size = 100000
    batch_size = 100
    log_every = 10

    with initialize(version_base=None, config_path="../scripts/config"):
        # overrides
        cfg = compose(
            config_name="config",
            overrides=[
                "mi_estimator=js",
                "data=multimixture",
                f"estimation.max_epochs=1",
                f"estimation.batch_size={batch_size}",
                "estimation.save_train_log=true",
                f"estimation.trained_model_save_path={model_path}",
                "estimation.valid_percentage=0.1",
                "estimation.early_stopping=false",
                f"data.n_samples={total_size}",
                "data.distribution.n_dim=1",
                f"logger.log_every={log_every}",
            ],
        )

        parse(cfg)

    estimator = torch.load(model_path)
    os.remove(model_path)

    p_xy = instantiate(cfg.data.distribution)
    true_mi = p_xy.mutual_information().item()
    test_samples = p_xy.sample(torch.Size([10000]))

    mi = evaluate_mi(
        estimator=estimator,
        data=test_samples,
        batch_size=64,
    )

    assert os.path.isfile("log.csv")
    loaded_train_log = pd.read_csv("log.csv")
    os.remove("log.csv")

    assert (
        len(loaded_train_log)
        == (total_size * 0.9) // batch_size // log_every * 2 + 2
    )

    print(mi, true_mi)
    assert np.isclose(mi, true_mi, 0.4)
