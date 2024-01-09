from typing import TypeVar, Optional, Union

from typing import List, Dict

import torch
from pyro.distributions import ConditionalDistribution
from torch.distributions import Distribution

from torch import nn

T = TypeVar("T", bound="NamedDistribution")


class JointDistribution(nn.Module, Distribution, ConditionalDistribution):
    def __init__(
        self,
        variables: List[str],
        name: str = "p",
    ):
        nn.Module.__init__(self)
        Distribution.__init__(self, validate_args=False)

        self.variables = variables
        self.name = name

    def full_name(self):
        return f"{self.name}({','.join(self.variables)})"

    def _log_prob(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def log_prob(self, *args, **kwargs) -> torch.Tensor:
        AMBIGUOUS_CALL_ERROR = "Please specify the variable names to avoid ambiguities. E.g. p.log_prob(x=values_x, y=values_y)"

        if len(args) > 1:
            raise ValueError(AMBIGUOUS_CALL_ERROR)
        elif len(args) == 1:
            if len(self.variables) > 1:
                raise ValueError(AMBIGUOUS_CALL_ERROR)
            elif self.variables[0] in kwargs:
                raise ValueError(AMBIGUOUS_CALL_ERROR)
            kwargs[self.variables[0]] = args[0]

        return self._log_prob(**kwargs)

    def _marginal(self: T, variables: List[str]) -> T:
        raise NotImplementedError()

    def marginal(self: T, *variables) -> T:
        valid_labels: List[str] = []
        for label in variables:
            if not (label in self.variables):
                raise ValueError(
                    f"Can not compute the marginal p({label}) since the distribution is defined on p({','.join(self.variables)})"
                )
            valid_labels.append(label)
        return self._marginal(valid_labels)

    def conditional(self, *variables) -> ConditionalDistribution:
        valid_variables: List[str] = []
        for variable in variables:
            if variable not in self.variables:
                raise ValueError(
                    f"The variable {variable} is not part of the support of {self.__repr__()}."
                )
            valid_variables.append(variable)
        return ConditionalNamedDistribution(self, valid_variables)

    def condition(self, **conditioning) -> T:
        valid_variables: List[str] = []
        for variable in conditioning:
            if variable not in self.variables:
                raise ValueError(
                    f"The variable {variable} is not part of the support of {self.__repr__()}."
                )
            valid_variables.append(variable)
        return ConditionedJointDistribution(
            joint=self,
            marginal=self.marginal(*valid_variables),
            cond_dict=conditioning,
        )

    def _mutual_information(
        self, variable_1: str, variable_2: str
    ) -> torch.Tensor:
        mi = (
            self.entropy(variable_1)
            + self.entropy(variable_2)
            - self.entropy(variable_1, variable_2)
        )
        return mi

    def mutual_information(
        self,
        variable_1: Optional[str] = None,
        variable_2: Optional[str] = None,
    ) -> torch.Tensor:
        if variable_1 is None:
            if not (variable_2) is None:
                raise ValueError(
                    "Please both variable for mutual information computation."
                )
            if not len(self.variables) == 2:
                raise ValueError(
                    "Please which variable should be used for mutual information computation."
                )
            variable_1 = self.variables[0]
            variable_2 = self.variables[1]

        if not (variable_1 in self.variables) or not (
            variable_2 in self.variables
        ):
            raise ValueError(
                f"Can not compute I({variable_1};{variable_2}) since the distribution is defined on p({','.join(self.variables)})"
            )
        return self._mutual_information(variable_1, variable_2)

    def _entropy(self, variables: List[str]) -> torch.Tensor:
        raise NotImplementedError()

    def entropy(self, *variables) -> torch.Tensor:
        valid_labels: List[str] = []

        if len(variables) == 0:
            if len(self.variables) > 1:
                raise ValueError(
                    f"Please specify the variables for the entropy computation among {self.variables} "
                )
            valid_labels = self.variables
        else:
            for variable in variables:
                assert isinstance(variable, str)
                if not (variable in self.variables):
                    raise ValueError(
                        f"Can not compute H({variables}) since the distribution is defined on p({','.join(self.variables)})"
                    )
                valid_labels.append(variable)
        return self._entropy(valid_labels)


class ConditionedJointDistribution(JointDistribution):
    def __init__(
        self,
        joint: JointDistribution,
        marginal: JointDistribution,
        cond_dict: Dict[str, torch.Tensor],
    ):
        if joint.name != marginal.name:
            raise ValueError("The two distribution must have the same name")

        super().__init__(
            name=joint.name,
            variables=[
                variable
                for variable in joint.variables
                if not (variable in cond_dict)
            ],
        )
        self.joint = joint
        self.log_marginal = marginal.log_prob(**cond_dict)
        self.cond_dict = cond_dict

    def _log_prob(self, **kwargs) -> torch.Tensor:
        kwargs = {**self.cond_dict, **kwargs}
        log_joint = self.joint.log_prob(**kwargs)
        log_conditional = log_joint - self.log_marginal
        return log_conditional

    def __repr__(self):
        s = super().__repr__()[:-1]
        s += f"|{','.join(self.cond_dict)})"
        return s


class ConditionalNamedDistribution(ConditionalDistribution):
    def __init__(
        self,
        joint: JointDistribution,
        condition_on: List[str],
    ):
        super().__init__()
        self.joint = joint
        self.condition_on = condition_on

    def condition(
        self, context: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> JointDistribution:
        if isinstance(context, torch.Tensor):
            assert len(self.condition_on) == 1
            context = {self.condition_on[0]: context}
        else:
            assert isinstance(context, dict)

        return ConditionedJointDistribution(
            joint=self.joint,
            marginal=self.joint.marginal(*self.condition_on),
            cond_dict=context,
        )

    def __repr__(self):
        s = f"{self.joint.name}({','.join([variable for variable in self.joint.variables if not (variable in self.condition_on)])}|"
        s += f"{','.join(self.condition_on)})"
        return s
