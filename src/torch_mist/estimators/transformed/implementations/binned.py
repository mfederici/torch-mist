from typing import Optional

from torch_mist.distributions.joint.categorical import JointCategorical
from torch_mist.estimators import GM
from torch_mist.estimators.transformed.base import TransformedMIEstimator
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.freeze import freeze


class BinnedMIEstimator(TransformedMIEstimator):
    # Technically this is not a lower-bound but the estimation of marginal entropy is usually accurate
    lower_bound = True

    def __init__(
        self,
        Q_x: Optional[QuantizationFunction] = None,
        Q_y: Optional[QuantizationFunction] = None,
        temperature: float = 1.0,
    ):
        freeze(Q_x)
        freeze(Q_y)

        q_XY = JointCategorical(
            variables=["x", "y"],
            bins=[Q_x.n_bins, Q_y.n_bins],
            temperature=temperature,
            name="q",
        )

        super().__init__(
            base_estimator=GM(
                q_XY=q_XY,
            ),
            transforms={
                "x": Q_x,
                "y": Q_y,
            },
        )
