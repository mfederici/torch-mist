from sklearn.datasets import load_iris
from torch_mist.decomposition import MID


def test_decomposition():
    # Load the Iris Dataset as a pandas DataFrame
    iris_dataset = load_iris(as_frame=True)
    data = iris_dataset["data"].values
    targets = iris_dataset["target"].values

    n_dim = 2

    for proj in [
        MID(n_dim),
        MID(n_dim, normalize_inputs=False),
        MID(n_dim, whiten=True, mi_estimator_params={"estimator_name": "nwj"}),
        MID(n_dim, proj_params={"hidden_dims": []}),
    ]:
        for train_params in [
            {},
            {"max_epochs": 200},
            {"optimizer_params": {"lr": 1e-4}},
            {"early_stopping": False},
        ]:
            z = proj.fit_transform(data, targets, **train_params)

            assert z.shape[0] == len(data)
            assert z.shape[1] == n_dim
            assert not (proj.train_log is None)
