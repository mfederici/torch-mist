import torch
from pyro.distributions import ConditionalDistribution

from torch_mist.distributions import CategoricalModule
from torch_mist.estimators import DoE
from torch_mist.estimators.transformed.base import TransformedMIEstimator
from torch_mist.quantization.functions import (
    QuantizationFunction,
    LearnableQuantization,
)


class PQ(TransformedMIEstimator):
    def __init__(
        self,
        q_QY_given_X: ConditionalDistribution,
        quantize_y: QuantizationFunction,
        temperature: float = 1.0,
    ):
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

        if (
            isinstance(quantize_y, LearnableQuantization)
            and not quantize_y.trained
        ):
            self._components_to_pretrain += [
                (lambda batch: batch["y"], self.transforms["y->y"])
            ]
