from typing import Optional, Union

from torch import nn

from torch_mist.utils.logging.logger.base import DummyLogger, Logger
from torch_mist.utils.logging.logger.pandas import PandasLogger


def instantiate_logger(
    model: nn.Module, logger: Optional[Union[bool, Logger]]
):
    # if logger is unspecified, use the default PandasLogger
    if logger is None:
        logger = PandasLogger()

    # if the logger is not logging anything, log the value of mutual information and loss by default
    if logger and not isinstance(logger, DummyLogger):
        # If no method is logged, add the loss and the mutual_information when available
        if len(logger._logged_methods) == 0:
            if hasattr(model, "mutual_information"):
                logger.log_method(model, "mutual_information")
            if hasattr(model, "loss"):
                logger.log_method(model, "loss")
    # if logger is False, we instantiate a dummy logger which does not log.
    else:
        logger = DummyLogger()

    return logger
