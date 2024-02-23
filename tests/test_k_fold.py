import numpy as np
from sklearn.datasets import load_iris
from torch_mist import k_fold_mi_estimate


def test_kfold_validation():
    iris_dataset = load_iris(as_frame=True)["data"]

    # Create np.arrays corresponding to petal and sepal size
    petal = iris_dataset[["petal length (cm)", "petal width (cm)"]].values
    sepal = iris_dataset[["sepal length (cm)", "sepal width (cm)"]].values

    true_value = 0.8

    for params in [
        dict(
            estimator="smile",  # Use the Smile mutual information estimator
            max_iterations=10000,  # Number of train iterations
            folds=5,  # Number of folds for cross-validation
            neg_samples=8,  # Number of negative samples
            seed=42,
            train_verbose=True,
        ),
        dict(
            estimator="nwj",  # Use the Smile mutual information estimator
            max_iterations=10000,  # Number of train iterations
            folds=5,  # Number of folds for cross-validation
            neg_samples=16,  # Number of negative samples
            hidden_dims=[256],
            seed=42,
        ),
    ]:
        # Estimate how much information the petal size and the target specie have in common
        estimated_mi, log = k_fold_mi_estimate(data=(petal, sepal), **params)

        assert np.isclose(estimated_mi, true_value, rtol=0.1), estimated_mi
        assert not (log is None)
