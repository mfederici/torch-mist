import os
from typing import Tuple, List, Callable, Optional

import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import random

from torch_mist.estimators import MIEstimator
from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid import HybridMIEstimator
from torch_mist.utils.data.utils import infer_dims, TensorDictLike
from torch_mist.utils.estimation import _instantiate_estimator, estimate_mi
from torch_mist.utils.logging.metrics import compute_mean_std


def default_logged_methods(
    estimator: MIEstimator,
) -> Tuple[List[Tuple[str, Callable]], List[Tuple[str, Callable]]]:
    eval_logged_methods = [
        ("log_ratio", compute_mean_std),
        ("batch_loss", compute_mean_std),
    ]

    if isinstance(estimator, DiscriminativeMIEstimator):
        eval_logged_methods += [
            ("approx_log_partition", compute_mean_std),
            ("unnormalized_log_ratio", compute_mean_std),
        ]

    if isinstance(estimator, HybridMIEstimator):
        eval_logged_methods += [
            ("generative_estimator.log_ratio", compute_mean_std),
            ("generative_estimator.batch_loss", compute_mean_std),
            ("discriminative_estimator.batch_loss", compute_mean_std),
            ("discriminative_estimator.log_ratio", compute_mean_std),
            (
                "discriminative_estimator.approx_log_partition",
                compute_mean_std,
            ),
            (
                "discriminative_estimator.unnormalized_log_ratio",
                compute_mean_std,
            ),
        ]

    train_logged_methods = [("batch_loss", compute_mean_std)]

    return train_logged_methods, eval_logged_methods


def prepare_data(
    conf: DictConfig,
) -> Tuple[
    TensorDictLike,
    Optional[TensorDictLike],
    Optional[TensorDictLike],
    Optional[float],
]:
    # Train (mandatory)
    data = instantiate(conf.data, _convert_="all")

    # Validation (optional)
    if "valid_data" in conf:
        valid_data = instantiate(conf.valid_data, _convert_="all")
    else:
        valid_data = None

    # Test (optional)
    if "test_data" in conf:
        test_data = instantiate(conf.test_data, _convert_="all")
    else:
        test_data = None

    # If the data is sampled from a distribution, compute the true mutual informaiton
    if "distribution" in conf:
        dist = instantiate(conf.distribution, _convert_="all")
        true_mi = dist.mutual_information()
    else:
        true_mi = None

    return data, valid_data, test_data, true_mi


@hydra.main(
    config_path="config", config_name="config.yaml", version_base="1.1"
)
def parse(conf: DictConfig):
    # Set the random seed
    if "seed" in conf:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)
        random.seed(conf.seed)

        if hasattr(conf.device, "tensor_cores"):
            torch.set_float32_matmul_precision(conf.device.matmul_precision)

    # Change the location to the original working directory (to fix issues with relative paths)
    hydra_wd = os.getcwd()
    try:
        os.chdir(hydra.utils.get_original_cwd())
    except Exception as e:
        print(e)
        pass

    # Instantiate the parameters and metadata if any
    if "params" in conf:
        conf.params = instantiate(conf.params)

    # Instantiate the loggers
    print("Instantiating the logger")
    logger = instantiate(conf.logger)

    # Add the configuration to the run if the logger allows
    if hasattr(logger, "add_config"):
        logger.add_config(OmegaConf.to_container(conf, resolve=True))

    # Instantiating the data
    print("Instantiating the Data")
    data, valid_data, test_data, true_mi = prepare_data(conf)

    # Infer the dimensionality of x and y if not specified by the config and update the values
    dims = infer_dims(data)
    conf.x_dim = dims[conf.estimation.x_key]
    conf.y_dim = dims[conf.estimation.y_key]

    # Instantiate the estimator
    print("Instantiating the Mutual Information Estimator")
    estimator = instantiate(conf.mi_estimator, _convert_="all")

    # Count and visualize the number of parameters
    n_parameters = 0
    for param in estimator.parameters():
        n_parameters += param.numel()
    print(f"{n_parameters} Parameters")

    # Define which quantities are logged
    train_logged_methods, eval_logged_methods = default_logged_methods(
        estimator
    )

    # Move back to the output directory created by hydra
    os.chdir(hydra_wd)

    # Train the estimator and evaluate the mutual information
    estimation_params = instantiate(conf.estimation, _convert_="all")
    estimated_mi, estimator, train_log = estimate_mi(
        estimator=estimator,
        data=data,
        valid_data=valid_data,
        test_data=test_data,
        logger=logger,
        train_logged_methods=train_logged_methods,
        eval_logged_methods=eval_logged_methods,
        return_estimator=True,
        **estimation_params,
    )

    if not (true_mi is None):
        print(f"True mi: {true_mi}")
    print(f"Estimated mi: {estimated_mi}")

    if conf.save_train_log:
        logger.save_log()

    if not (conf.save_trained_model is None):
        print(f"Saving the estimator in {conf.save_trained_model}")
        logger.save_model(estimator, conf.save_trained_model)


def run():
    OmegaConf.register_new_resolver("eval", eval)
    parse()


if __name__ == "__main__":
    run()
