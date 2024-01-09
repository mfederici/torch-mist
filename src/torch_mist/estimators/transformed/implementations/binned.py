from typing import Optional

from torch_mist.distributions.joint.categorical import JointCategorical
from torch_mist.estimators.generative.base import JointGenerativeMIEstimator
from torch_mist.estimators.transformed.base import TransformedMIEstimator
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.freeze import freeze


class BinnedMIEstimator(TransformedMIEstimator):
    # Technically this is not a lower-bound but the estimation of marginal entropy is usually accurate
    lower_bound = True

    def __init__(
        self,
        quantize_x: Optional[QuantizationFunction] = None,
        quantize_y: Optional[QuantizationFunction] = None,
        temperature: float = 1.0,
    ):
        quantize_x = freeze(quantize_x)
        quantize_y = freeze(quantize_y)

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
