from typing import Optional, Union

from torch_mist.estimators.base import MIEstimator
from torch_mist.utils.logging.logger.base import DummyLogger, Logger
from torch_mist.utils.logging.logger.pandas import PandasLogger


def instantiate_mi_logger(
    estimator: MIEstimator, logger: Optional[Union[bool, Logger]]
):
    # if logger is unspecified, use the default PandasLogger
    if logger is None:
        logger = PandasLogger()

    # if the logger is not logging anything, log the value of mutual information and loss by default
    if logger and not isinstance(logger, DummyLogger):
        # If no method is logged, add the loss and the expected_log_ratio
        if len(logger._logged_methods) == 0:
            logger.log_method(estimator, "mutual_information")
            logger.log_method(estimator, "loss")
    # if logger is False, we instantiate a dummy logger which does not log.
    else:
        logger = DummyLogger()

    return logger
