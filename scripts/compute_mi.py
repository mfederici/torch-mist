import numpy as np
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import random

from torch_mist.estimators.discriminative import DiscriminativeMIEstimator
from torch_mist.estimators.hybrid import HybridMIEstimator
from torch_mist.utils import train_mi_estimator, evaluate_mi
from torch_mist.utils.logging.metrics import compute_mean_std


@hydra.main(config_path="config", config_name="config.yaml")
def parse(conf: DictConfig):
    # Instatiate the parameters and metadata if any
    if "params" in conf:
        conf.params = instantiate(conf.params)

    # Instantiate the loggers
    print("Instantiating the logger")
    logger = instantiate(conf.logger)

    # Add the configuration to the run if the logger allows
    if hasattr(logger, "add_config"):
        logger.add_config(OmegaConf.to_container(conf, resolve=False))

    # Set the random seed
    if "seed" in conf:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)
        random.seed(conf.seed)

        if hasattr(conf.device, "tensor_cores"):
            torch.set_float32_matmul_precision(conf.device.matmul_precision)

    # Instantiating the distribution
    print("Instantiating the Distributions")
    train_samples = instantiate(conf.data.train, _convert_="all")
    test_samples = instantiate(conf.data.test, _convert_="all")
    if hasattr(conf.data, "distribution"):
        dist = instantiate(conf.data.distribution, _convert_="all")
        true_mi = dist.mutual_information()
    else:
        true_mi = None

    # Add the sampled x and y as parameters of the quantization scheme
    extra_params = {}
    if hasattr(conf.mi_estimator, "quantize_x"):
        print("Training the quantization for x")
        extra_params["quantize_x"] = instantiate(
            conf.mi_estimator.quantize_x,
            _convert_="all",
            data=train_samples["x"],
        )
    if hasattr(conf.mi_estimator, "quantize_y"):
        print("Training the quantization for y")
        extra_params["quantize_y"] = instantiate(
            conf.mi_estimator.quantize_y,
            _convert_="all",
            data=train_samples["y"],
        )

    # Instantiate the estimator
    print("Instantiating the Mutual Information Estimator")
    mi_estimator = instantiate(
        conf.mi_estimator, _convert_="all", **extra_params
    )
    print(mi_estimator)

    n_parameters = 0
    for param in mi_estimator.parameters():
        n_parameters += param.numel()
    print(f"{n_parameters} Parameters")

    logged_methods = [
        ("log_ratio", compute_mean_std),
        ("batch_loss", compute_mean_std),
    ]

    if isinstance(mi_estimator, DiscriminativeMIEstimator):
        logged_methods += [
            ("approx_log_partition", compute_mean_std),
            ("unnormalized_log_ratio", compute_mean_std),
        ]

    if isinstance(mi_estimator, HybridMIEstimator):
        logged_methods += [
            ("generative_estimator.log_ratio", compute_mean_std),
            ("discriminative_estimator.log_ratio", compute_mean_std),
            ("generative_estimator.batch_loss", compute_mean_std),
            ("discriminative_estimator.batch_loss", compute_mean_std),
        ]

    with logger.logged_methods(mi_estimator, logged_methods):
        train_mi_estimator(
            mi_estimator,
            x=train_samples["x"],
            y=train_samples["y"],
            logger=logger,
            **conf.params.train,
        )

        with logger.test():
            results = evaluate_mi(
                mi_estimator,
                x=test_samples["x"],
                y=test_samples["y"],
                **conf.params.test,
            )

        if not (true_mi is None):
            print(f"True mi: {true_mi}")
        print(f"Estimated mi: {results}")
        logger.save_log()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    parse()
