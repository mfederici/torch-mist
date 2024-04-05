import os
from typing import Tuple, List, Callable, Optional

import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import random

from torch_mist.estimators.hybrid.base import HybridMIEstimator
from torch_mist.estimators.discriminative.base import DiscriminativeMIEstimator
from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.logging.metrics import compute_mean_std


def advanced_logged_methods(
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


def default_logged_methods(
    estimator: MIEstimator,
) -> Tuple[List[Tuple[str, Callable]], List[Tuple[str, Callable]]]:
    eval_logged_methods = [
        ("log_ratio", compute_mean_std),
    ]

    train_logged_methods = [
        ("batch_loss", compute_mean_std),
        ("log_ratio", compute_mean_std),
    ]

    return train_logged_methods, eval_logged_methods


@hydra.main(
    config_path="config", config_name="config.yaml", version_base="1.1"
)
def parse(conf: DictConfig):
    # Set the random seed
    if "seed" in conf:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)
        random.seed(conf.seed)

        if hasattr(conf.hardware, "tensor_cores"):
            if conf.hardware.tensor_cores:
                torch.set_float32_matmul_precision(
                    conf.hardware.matmul_precision
                )

    # Change the location to the original working directory (to fix issues with relative paths)
    hydra_wd = os.getcwd()
    try:
        os.chdir(hydra.utils.get_original_cwd())
    except Exception as e:
        print(e)
        pass

    # Instantiate the logger
    logger = instantiate(conf.logger, _convert_="all")
    if hasattr(logger, "add_config"):
        logger.add_config(OmegaConf.to_container(conf, resolve=True))

    # Instantiating the data
    print("Instantiating the Data")
    data = instantiate(conf.data, _convert_="all")

    # Move back to the output directory created by hydra
    os.chdir(hydra_wd)

    estimation = instantiate(conf.estimation, _partial_=True)

    # Train the estimator and evaluate the mutual information
    mi_estimate, log = estimation(logger=logger, data=data)

    print(f"Estimated mi: {mi_estimate}")


def run():
    OmegaConf.register_new_resolver("eval", eval)
    parse()


if __name__ == "__main__":
    run()
