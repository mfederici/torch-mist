import numpy as np
import torch
from torch.utils.data import DataLoader

from torch_mist import estimate_mi
from torch_mist.utils.batch import unfold_samples
from torch_mist.utils.data import SampleDataset


def test_inputs_train_mi_estimator():
    x = torch.zeros(100, 1)
    y = torch.zeros(100, 1)
    x.normal_()
    y.normal_()

    wrong_y = torch.zeros([101, 1])

    dataset = SampleDataset({"x": x, "y": y})

    # Then we make a DataLoader
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    valid_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    test_loader = valid_loader

    wrong_train_loader = DataLoader(SampleDataset({"w": x, "y": y}))

    tests = [
        {
            "params": {"x": x, "y": wrong_y, "max_epochs": 1},
            "should_fail": True,
            "message": "The estimator should fail when passed x and y of different length.",
        },
        {
            "params": {"x": x, "max_epochs": 1},
            "should_fail": True,
            "message": "The estimator should fail when only x is passed.",
        },
        {
            "params": {"y": y, "max_epochs": 1},
            "should_fail": True,
            "message": "The estimator should fail when only y is passed.",
        },
        {
            "params": {"x": x, "y": y, "max_epochs": 1},
            "should_fail": False,
            "message": "Failed with x,y and max_epochs",
        },
        {
            "params": {"x": x, "y": y, "max_iterations": 1},
            "should_fail": False,
            "message": "Failed with x,y and max_iterations",
        },
        {
            "params": {"x": x, "y": y},
            "should_fail": True,
            "message": "Failed with x,y",
        },
        {
            "params": {"train_loader": train_loader, "max_epochs": 1},
            "should_fail": False,
            "message": "Failed with train_loader only",
        },
        {
            "params": {"train_loader": wrong_train_loader, "max_epochs": 1},
            "should_fail": True,
            "message": "The function should fail when a non-valid train_loader is passed",
        },
        {
            "params": {
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with train and valid_loader",
        },
        {
            "params": {
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "test_loader": test_loader,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with all the loaders",
        },
        {
            "params": {
                "train_loader": train_loader,
                "test_loader": test_loader,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with train and test_loader",
        },
        {
            "params": {
                "valid_loader": valid_loader,
                "test_loader": test_loader,
                "max_epochs": 1,
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
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with x,y, batch and test_loader",
        },
    ]

    for test in tests:
        print(
            {
                k: v.shape if isinstance(v, torch.Tensor) else v
                for k, v in test["params"].items()
            }
        )

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
