# read version from installed package
from importlib.metadata import version

__version__ = version("torch_mist")
from .utils.estimation import estimate_mi
