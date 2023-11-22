import numpy as np
import torch

from torch_mist import estimate_mi
from torch.utils.data import DataLoader

from torch_mist.utils.data import SampleDataset
from torch_mist.utils.batch import unfold_samples


def test_inputs_train_mi_estimator():
    x = np.zeros([100, 1]).astype(np.float32)
    y = np.zeros([100, 1]).astype(np.float32)
    wrong_y = np.zeros([101, 1]).astype(np.float32)

    dataset = SampleDataset({"x": x, "y": y})

    # Then we make a DataLoader
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    valid_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    test_loader = valid_loader

    tests = [
        {
            "params": {"x": x, "y": wrong_y},
            "should_fail": True,
            "message": "The estimator should fail when passed x and y of different length.",
        },
        {
            "params": {"x": x, "x_dim": 1, "y_dim": 1},
            "should_fail": True,
            "message": "The estimator should fail when only x is passed.",
        },
        {
            "params": {"y": y, "x_dim": 1, "y_dim": 1},
            "should_fail": True,
            "message": "The estimator should fail when only y is passed.",
        },
        {
            "params": {"x": x, "y": y},
            "should_fail": False,
            "message": "Failed with x,y",
        },
        {
            "params": {"train_loader": train_loader, "x_dim": 1, "y_dim": 1},
            "should_fail": False,
            "message": "Failed with train_loader only",
        },
        {
            "params": {
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "x_dim": 1,
                "y_dim": 1,
            },
            "should_fail": False,
            "message": "Failed with train and valid_loader",
        },
        {
            "params": {
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "test_loader": test_loader,
                "x_dim": 1,
                "y_dim": 1,
            },
            "should_fail": False,
            "message": "Failed with all the loaders",
        },
        {
            "params": {
                "train_loader": train_loader,
                "test_loader": test_loader,
                "x_dim": 1,
                "y_dim": 1,
            },
            "should_fail": False,
            "message": "Failed with train and test_loader",
        },
        {
            "params": {
                "valid_loader": valid_loader,
                "test_loader": test_loader,
                "x_dim": 1,
                "y_dim": 1,
            },
            "should_fail": True,
            "message": "A train_loader should be provided",
        },
        {
            "params": {
                "x": x,
                "y": y,
                "batch_size": 10,
                "test_loader": test_loader,
            },
            "should_fail": False,
            "message": "Failed with x,y, batch and test_loader",
        },
    ]

    for test in tests:
        failed = False
        try:
            estimate_mi(
                estimator_name="js", hidden_dims=[32, 32], **test["params"]
            )
        except Exception as e:
            failed = True
            print(e)
        assert failed == test["should_fail"], test["message"]


def test_utils():
    x = torch.zeros(10, 1)
    failed = False
    try:
        unfold_samples((x,))
    except Exception as e:
        print(e)
        failed = True

    assert failed

    failed = False
    try:
        unfold_samples({"x": x})
    except Exception as e:
        print(e)
        failed = True

    assert failed

    failed = False
    try:
        unfold_samples(x)
    except Exception as e:
        print(e)
        failed = True

    assert failed
