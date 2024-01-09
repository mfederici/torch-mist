import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.distributions import CategoricalModule
from torch_mist.estimators import DoE
from torch_mist.estimators.transformed.base import TransformedMIEstimator
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.freeze import freeze


class PQ(TransformedMIEstimator):
    def __init__(
        self,
        q_QY_given_X: ConditionalDistribution,
        quantize_y: QuantizationFunction,
        temperature: float = 1.0,
    ):
        quantize_y = freeze(quantize_y)

        super().__init__(
            transforms={"y": quantize_y},
            base_estimator=DoE(
                q_Y_given_X=q_QY_given_X,
                q_Y=CategoricalModule(
                    logits=torch.zeros(quantize_y.n_bins),
                    temperature=temperature,
                ),
            ),
        )
