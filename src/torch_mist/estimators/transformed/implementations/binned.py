from typing import Optional

from torch_mist.distributions.joint.categorical import JointCategorical
from torch_mist.estimators.generative.base import JointGenerativeMIEstimator
from torch_mist.estimators.transformed.base import TransformedMIEstimator
from torch_mist.quantization.functions import (
    QuantizationFunction,
    LearnableQuantization,
)


class BinnedMIEstimator(TransformedMIEstimator):
    # Technically this is not a lower-bound but the estimation of marginal entropy is usually accurate
    lower_bound = True

    def __init__(
        self,
        quantize_x: Optional[QuantizationFunction] = None,
        quantize_y: Optional[QuantizationFunction] = None,
        temperature: float = 1.0,
    ):
        q_XY = JointCategorical(
            variables=["x", "y"],
            bins=[quantize_x.n_bins, quantize_y.n_bins],
            temperature=temperature,
            name="q",
        )

        transforms = {}
        if quantize_x:
            transforms["x"] = quantize_x
        if quantize_y:
            transforms["y"] = quantize_y

        super().__init__(
            base_estimator=JointGenerativeMIEstimator(
                q_XY=q_XY,
            ),
            transforms=transforms,
        )

        if (
            isinstance(quantize_y, LearnableQuantization)
            and not quantize_y.trained
        ):
            self._components_to_pretrain += [
                (lambda batch: batch["y"], self.transforms["y->y"])
            ]

        if (
            isinstance(quantize_y, LearnableQuantization)
            and not quantize_y.trained
        ):
            self._components_to_pretrain += [
                (lambda batch: batch["x"], self.transforms["x->x"])
            ]
