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
        conf.params = instantiate(conf.params, _convert_="all")

    # Instantiate the loggers
    print("Instantiating the logger")
    logger = instantiate(conf.logger)

    # Set the random seed
    if "seed" in conf:
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)
        random.seed(conf.seed)

        if hasattr(conf.device, "tensor_cores"):
            torch.set_float32_matmul_precision(conf.device.matmul_precision)

    # Instantiating the samples
    print("Instantiating the Distributions")
    distribution = instantiate(conf.distribution, _convert_="all")
    n_train = instantiate(conf.n_train, _convert_="all")
    n_test = instantiate(conf.n_test, _convert_="all")

    train_samples = distribution.sample(n_train)
    test_samples = distribution.sample(n_test)

    # Instantiate the estimator
    estimator = instantiate(conf.mi_estimator, _convert_="all")

    logged_methods = [
        ("log_ratio", compute_mean_std),
        ("unnormalized_log_ratio", compute_mean_std),
        ("approx_log_partition", compute_mean_std),
        ("batch_loss", compute_mean_std),
    ]

    if isinstance(estimator, DiscriminativeMIEstimator):
        logged_methods += [
            ("unnormalized_log_ratio", compute_mean_std),
            ("approx_log_partition", compute_mean_std),
        ]

    if isinstance(estimator, HybridMIEstimator):
        logged_methods.append(
            ("generative_estimator.log_ratio", compute_mean_std)
        )

    with logger.logged_methods(logged_methods):
        train_mi_estimator(
            estimator,
            x=train_samples["x"],
            y=train_samples["y"],
            **conf.params.train,
        )

        with logger.test():
            results = evaluate_mi(
                estimator,
                x=test_samples["x"],
                y=test_samples["y"],
                **conf.params.test,
            )
        print(results)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    parse()
