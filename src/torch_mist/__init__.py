# read version from installed package
from importlib.metadata import version

__version__ = version("torch_mist")
from torch_mist.utils.estimation import estimate_mi
