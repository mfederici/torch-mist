from abc import abstractmethod
from typing import Optional, Union

import torch
from pyro.distributions import ConditionalDistribution
from torch import nn
from torch.distributions import Distribution

from torch_mist.distributions.conditional.delta import ConditionalDelta
from torch_mist.estimators import MIEstimator
from torch_mist.nn import Identity, Model


class InformationBottleneck(Model):
    def __init__(
        self,
        mi_estimator: MIEstimator,
        beta: float,
        p_ZX_given_X: Optional[
            Union[nn.Module, ConditionalDistribution]
        ] = None,
        p_ZY_given_Y: Optional[
            Union[nn.Module, ConditionalDistribution]
        ] = None,
    ):
        super().__init__()
        self.upper_bound = mi_estimator.lower_bound

        self.beta = beta
        self.mi_estimator = mi_estimator

        if p_ZX_given_X is None:
            p_ZX_given_X = Identity()
        if p_ZY_given_Y is None:
            p_ZY_given_Y = Identity()

        if not isinstance(p_ZX_given_X, ConditionalDistribution):
            p_ZX_given_X = ConditionalDelta(p_ZX_given_X)
        if not isinstance(p_ZY_given_Y, ConditionalDistribution):
            p_ZY_given_Y = ConditionalDelta(p_ZY_given_Y)

        self.encoder_x = p_ZX_given_X
        self.encoder_y = p_ZY_given_Y

    @abstractmethod
    def regularization(
        self,
        zx: torch.Tensor,
        zy: torch.Tensor,
        p_ZX_given_x: Distribution,
        p_ZY_given_y: Distribution,
    ) -> torch.Tensor():
        raise NotImplementedError()

    def mutual_information(self, x: torch.Tensor, y: torch.Tensor):
        return self.mi_estimator.mutual_information(x, y)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p_zx_given_x = self.encoder_x.condition(x)
        p_zy_given_y = self.encoder_y.condition(y)
        zx = p_zx_given_x.rsample()
        zy = p_zy_given_y.rsample()

        mi_loss = self.mi_estimator.loss(zx, zy)
        reg_loss = self.regularization(zx, zy, p_zx_given_x, p_zy_given_y)
        assert reg_loss.numel() == 1

        return mi_loss + self.beta * reg_loss


class VIB(InformationBottleneck):
    def __init__(
        self,
        mi_estimator: MIEstimator,
        q_ZX: Distribution,
        beta: float,
        p_ZX_given_X: Optional[
            Union[nn.Module, ConditionalDistribution]
        ] = None,
        p_ZY_given_Y: Optional[
            Union[nn.Module, ConditionalDistribution]
        ] = None,
    ):
        super().__init__(
            mi_estimator=mi_estimator,
            beta=beta,
            p_ZX_given_X=p_ZX_given_X,
            p_ZY_given_Y=p_ZY_given_Y,
        )
        self.q_ZX = q_ZX

    def regularization(
        self,
        zx: torch.Tensor,
        zy: torch.Tensor,
        p_ZX_given_x: Distribution,
        p_ZY_given_y: Distribution,
    ) -> torch.Tensor():
        KL = (p_ZX_given_x.log_prob(zx) - self.q_ZX.log_prob(zx)).mean()

        return KL


class MIB(InformationBottleneck):
    def regularization(
        self,
        zx: torch.Tensor,
        zy: torch.Tensor,
        p_ZX_given_x: Distribution,
        p_ZY_given_y: Distribution,
    ) -> torch.Tensor():
        assert (
            zx.shape == zy.shape
        ), f"MIB requires representations of the same shape, {zx.shape}!={zy.shape}"

        KL_x = p_ZX_given_x.log_prob(zx) - p_ZY_given_y.log_prob(zx)
        KL_y = p_ZY_given_y.log_prob(zy) - p_ZX_given_x.log_prob(zy)

        return (KL_x + KL_y).mean() / 2.0


class CEB(InformationBottleneck):
    def __init__(
        self,
        mi_estimator: MIEstimator,
        q_ZX_given_ZY: ConditionalDistribution,
        beta: float,
        p_ZX_given_X: Optional[
            Union[nn.Module, ConditionalDistribution]
        ] = None,
        p_ZY_given_Y: Optional[
            Union[nn.Module, ConditionalDistribution]
        ] = None,
    ):
        super().__init__(
            mi_estimator=mi_estimator,
            beta=beta,
            p_ZX_given_X=p_ZX_given_X,
            p_ZY_given_Y=p_ZY_given_Y,
        )
        self.q_ZX_given_ZY = q_ZX_given_ZY

    def regularization(
        self,
        zx: torch.Tensor,
        zy: torch.Tensor,
        p_ZX_given_x: Distribution,
        p_ZY_given_y: Distribution,
    ) -> torch.Tensor():
        q_ZX_given_zy = self.q_ZX_given_ZY.condition(zy)
        KL = (p_ZX_given_x.log_prob(zx) - q_ZX_given_zy.log_prob(zx)).mean()

        return KL
