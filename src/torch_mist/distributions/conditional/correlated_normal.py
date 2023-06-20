import torch
from pyro.distributions import ConditionalDistribution, Independent
from torch.distributions import Normal, Transform, TransformedDistribution, HalfNormal
from torch.distributions import constraints


class CorrelatedNormal(ConditionalDistribution):
    def __init__(self, rho: float):
        self.rho = rho

    def condition(self, context):
        p_y_x = Independent(Normal(loc=context*self.rho, scale=(1-self.rho**2)**0.5), 1)
        return p_y_x


class CubicPTransform(Transform):
    bijective = True
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, k1: float = 1.0, k3: float = 1.0):
        super(CubicPTransform, self).__init__()
        self.k1 = k1
        self.k3 = k3

    def _call(self, x):
        return self.k1 * x + self.k3*x ** 3

    def _inverse(self, y):
        c1 = (3 * (27 * self.k3**4 * y ** 2 + 4 * self.k3**3 * self.k1 ** 3)) ** (1 / 2) - 9 * self.k3**2 * y
        c = c1.abs()**(1.0/3.0) * ((c1 >= 0).float() * 2.0 - 1.0)
        c2 = (2.0/3.0)**(1.0/3.0)*self.k1
        c3 = (2*9)**(1.0/3.0)*self.k3
        x = c2/c-c/c3
        return x

    def log_abs_det_jacobian(self, x, y):
        return (3 * self.k3 * x ** 2 + self.k1).log().sum(-1)


class RandomFlipTransform(Transform):
    bijective = False
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def _call(self, x):
        b = (torch.rand(x.shape)>0.5).float().to(x.device)*2 -1
        return b*x


class CubicCorrelatedNormal(ConditionalDistribution):
    def __init__(self, rho: float):
        self.rho = rho

    def condition(self, context):
        p_y_x = Independent(Normal(loc=(context*self.rho), scale=(1-self.rho**2)**0.5), 1)
        return TransformedDistribution(p_y_x, [CubicPTransform(k1=0.5, k3=2), RandomFlipTransform()])


class Translate(Transform):
    bijective = True
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, b):
        super(Translate, self).__init__()
        self.b = b

    def _call(self, x):
        return x+self.b

    def _inverse(self, y):
        return y-self.b

    def log_abs_det_jacobian(self, x, y):
        return x.sum(-1)*0


class SkewedCorrelatedNormal(ConditionalDistribution):
    def __init__(self, rho: float):
        self.rho = rho

    def condition(self, context):
        p_y_x = Independent(HalfNormal(scale=context*0+(1-self.rho**2)**0.5), 1)
        translate = Translate(context * self.rho)

        return TransformedDistribution(p_y_x, [translate])

# class CorrelatedNormal(Normal):
#     support = constraints.real_vector
#
#     def __init__(self, dim: int, rho: float):
#         loc = torch.zeros(dim*2)
#         super(CorrelatedNormal, self).__init__(loc=loc, scale=1)
#         self.rho = rho
#
#     def sample(self, sample_shape=torch.Size()):
#         s = super(CorrelatedNormal, self).sample(sample_shape)
#         x, y = torch.split(s, s.shape[-1]//2, -1)
#         y = self.rho*x + (1.0 - self.rho**2) ** 0.5 * y
#         return torch.cat([
#             x.unsqueeze(-1),
#             y.unsqueeze(-1),
#         ], -1)