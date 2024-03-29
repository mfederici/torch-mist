from sklearn.datasets import load_iris
from torch_mist.decomposition import MID, VIB, MIB, CEB, TIB, TInfoMax
import torch


def test_decomposition():
    # Load the Iris Dataset as a pandas DataFrame
    iris_dataset = load_iris(as_frame=True)
    data = iris_dataset["data"].values
    targets = iris_dataset["target"].values

    n_dim = 2

    for proj in [
        MID(n_dim),
        MID(n_dim, normalize_inputs=False),
        MID(n_dim, whiten=True, model_params={"estimator_name": "nwj"}),
        MID(n_dim, proj_params={"hidden_dims": []}),
        VIB(n_dim, beta=0.1),
        MIB(n_dim, stochastic_transform=True),
        CEB(n_dim, whiten=True),
    ]:
        for train_params in [
            {"max_epochs": 100},
            {"optimizer_params": {"lr": 1e-4}},
        ]:
            print(proj)
            z = proj.fit_transform(data, targets, **train_params)

            assert z.shape[0] == len(data)
            assert z.shape[1] == n_dim
            assert not (proj.train_log is None)


def test_temporal_decompostion():
    samples = torch.zeros(1000)
    samples.normal_()
    samples *= 0.1

    # Brownian trajectory
    traj = torch.cumsum(samples, 0).unsqueeze(-1)
    n_dim = 1

    for proj in [
        TIB(n_dim, beta=0.01, lagtime=10),
        TInfoMax(n_dim, normalize_inputs=False, lagtime=20),
    ]:
        for train_params in [
            {"max_epochs": 100},
            {"optimizer_params": {"lr": 1e-4}},
        ]:
            print(proj)
            z = proj.fit_transform(traj, **train_params)

            assert z.shape[0] == len(traj)
            assert z.shape[1] == n_dim
            assert not (proj.train_log is None)
