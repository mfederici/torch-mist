from pyro.distributions import ConditionalDistribution

from torch_mist.distributions.joint.categorical import JointCategorical
from torch_mist.estimators import DoE
from torch_mist.estimators.transformed.base import TransformedMIEstimator
from torch_mist.quantization.functions import QuantizationFunction
from torch_mist.utils.freeze import freeze


class PQ(TransformedMIEstimator):
    def __init__(
        self,
        q_QY_given_X: ConditionalDistribution,
        Q_y: QuantizationFunction,
        temperature: float = 1.0,
    ):
        freeze(Q_y)

        super().__init__(
            transforms={"y": Q_y},
            base_estimator=DoE(
                q_Y_given_X=q_QY_given_X,
                q_Y=JointCategorical(
                    variables=["y"],
                    bins=[Q_y.n_bins],
                    temperature=temperature,
                ),
            ),
        )
