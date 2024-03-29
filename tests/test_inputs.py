import torch
from torch.utils.data import DataLoader

from torch_mist import estimate_mi
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
            "params": {"data": {"x": x, "y": y}, "max_epochs": 1},
            "should_fail": False,
            "message": "Failed with x,y and max_epochs",
        },
        {
            "params": {"data": (x, wrong_y), "max_epochs": 1},
            "should_fail": True,
            "message": "The estimator should fail when passed x and y of different length.",
        },
        {
            "params": {"data": x, "max_epochs": 1},
            "should_fail": True,
            "message": "The estimator should fail when only x is passed.",
        },
        {
            "params": {"data": y, "max_epochs": 1},
            "should_fail": True,
            "message": "The estimator should fail when only y is passed.",
        },
        {
            "params": {"data": (x, y), "max_epochs": 1},
            "should_fail": False,
            "message": "Failed with x,y and max_epochs",
        },
        {
            "params": {"data": {"x": x, "y": y}, "max_iterations": 1},
            "should_fail": False,
            "message": "Failed with x,y and max_iterations",
        },
        {
            "params": {"data": {"x": x, "y": y}},
            "should_fail": False,
            "message": "Failed with x,y",
        },
        {
            "params": {
                "data": train_loader,
                "max_epochs": 1,
                "valid_percentage": 0.1,
            },
            "should_fail": False,
            "message": "Failed with train_loader, max_epochs and valid_percentage>0.",
        },
        {
            "params": {
                "data": train_loader,
                "max_epochs": 1,
                "valid_percentage": 0,
            },
            "should_fail": False,
            "message": "Failed with train_loader only",
        },
        {
            "params": {"data": wrong_train_loader, "max_epochs": 1},
            "should_fail": True,
            "message": "The function should fail when a non-valid train_loader is passed",
        },
        {
            "params": {
                "data": train_loader,
                "valid_data": valid_loader,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with train and valid_loader",
        },
        {
            "params": {
                "data": dataset,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with train_set",
        },
        {
            "params": {
                "data": train_loader,
                "valid_data": valid_loader,
                "test_data": test_loader,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with all the loaders",
        },
        {
            "params": {
                "data": train_loader,
                "test_data": test_loader,
                "valid_percentage": 0.1,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with train_loader and test_loader.",
        },
        {
            "params": {
                "data": train_loader,
                "test_data": test_loader,
                "valid_percentage": 0,
                "max_epochs": 1,
            },
            "should_fail": False,
            "message": "Failed with train_loader, test_loader and no valid_loader and valid_percentage=0.",
        },
        {
            "params": {
                "valid_data": valid_loader,
                "test_data": test_loader,
                "max_epochs": 1,
            },
            "should_fail": True,
            "message": "A train_loader should be provided",
        },
        {
            "params": {
                "data": {
                    "x": x,
                    "y": y,
                },
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
                k: v.shape if hasattr(v, "shape") else v
                for k, v in test["params"].items()
            }
        )

        failed = False
        try:
            estimate_mi(
                estimator_name="js", hidden_dims=[32, 32], **test["params"]
            )
        except Exception as e:
            print("##################")
            print(e)
            failed = True
        assert failed == test["should_fail"], test["message"]
