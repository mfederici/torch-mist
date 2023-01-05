from torch import nn

from core.models import MutualInformationEstimator  # TODO integrate with the rest
from core.models.baseline import LearnableJointBaseline, ConstantBaseline


class SimCLR(MutualInformationEstimator):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        MutualInformationEstimator.__init__(
            self,
            *args,
            baseline=ConstantBaseline(0),
            **kwargs
        )

    @staticmethod
    def _compute_dual_ratio_value(x, y, f, f_, baseline):
        b = baseline(f_, x, y)
        if b.ndim == 1:
            b = b.unsqueeze(1)
        assert b.ndim == f_.ndim

        Z = f_.exp().mean(1).unsqueeze(1) / (f - b).exp()

        ratio_value = b - Z + 1
        return ratio_value.mean()
