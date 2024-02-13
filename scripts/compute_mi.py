import os

import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import random

from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid import HybridMIEstimator
from torch_mist.utils import train_mi_estimator, evaluate_mi
from torch_mist.utils.estimation import _instantiate_estimator, estimate_mi
from torch_mist.utils.logging.metrics import compute_mean_std


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
    os.chdir(hydra.utils.get_original_cwd())

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

    # Handle estimating mutual information between variables different from 'x' and 'y'
    if "x_key" in conf:
        x_key = conf.x_key
    else:
        x_key = None
    if "y_key" in conf:
        y_key = conf.y_key
    else:
        y_key = None

    # Add the sampled x and y as parameters of the quantization scheme
    # TODO this does not work for multi-estimators and when datasets are passed
    extra_params = {}
    if hasattr(conf.mi_estimator, "quantize_x"):
        print(f"Training the quantization for {x_key}")
        extra_params["quantize_x"] = instantiate(
            conf.mi_estimator.quantize_x,
            _convert_="all",
            data=data[x_key],
        )
    if hasattr(conf.mi_estimator, "quantize_y"):
        print("Training the quantization for y")
        extra_params["quantize_y"] = instantiate(
            conf.mi_estimator.quantize_y,
            _convert_="all",
            data=data[y_key],
        )

    # Instantiate the estimator
    print("Instantiating the Mutual Information Estimator")

    estimator = _instantiate_estimator(
        instantiate,
        data=data,
        config=conf.mi_estimator,
        x_key=x_key,
        y_key=y_key,
        verbose=True,
        _convert_="all",
        **extra_params,
    )

    n_parameters = 0
    for param in estimator.parameters():
        n_parameters += param.numel()
    print(f"{n_parameters} Parameters")

    # Define which quantities are logged
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

    # Move back to the output directory created by hydra
    os.chdir(hydra_wd)

    # Train the estimator and evaluate the mutual information
    estimated_mi, train_log = estimate_mi(
        estimator=estimator,
        data=data,
        valid_data=valid_data,
        test_data=test_data,
        logger=logger,
        train_logged_methods=train_logged_methods,
        eval_logged_methods=eval_logged_methods,
        return_estimator=False,
        **conf.estimation,
    )

    if not (true_mi is None):
        print(f"True mi: {true_mi}")
    print(f"Estimated mi: {estimated_mi}")

    logger.save_log()

    if conf.save_trained_model:
        logger.save_model(estimator, "mi_estimator.pyt")


def run():
    OmegaConf.register_new_resolver("eval", eval)
    parse()


if __name__ == "__main__":
    run()
