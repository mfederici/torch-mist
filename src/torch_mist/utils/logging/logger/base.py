from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Optional, Union, List, Tuple, Dict

from torch_mist.utils.logging.contexts import (
    LoggingContext,
    IncrementalLoggingContext,
    MethodsLoggingContext,
)
from torch_mist.utils.misc import args_to_kwargs


class Logger:
    def __init__(self):
        self._logged_methods = {}
        self._context = {}

    def context(self, context_name: str, context_value: Any) -> LoggingContext:
        return LoggingContext(self, context_name, context_value)

    def train(self) -> LoggingContext:
        return self.context("split", "train")

    def valid(self) -> LoggingContext:
        return self.context("split", "valid")

    def test(self) -> LoggingContext:
        return self.context("split", "test")

    def iteration(self) -> LoggingContext:
        return IncrementalLoggingContext(self, "iteration")

    def epoch(self) -> LoggingContext:
        return IncrementalLoggingContext(self, "epoch")

    def logged_methods(
        self,
        instance: Any,
        methods: List[Union[str, Tuple[str, Callable[[Any, Any], Any]]]],
    ) -> MethodsLoggingContext:
        return MethodsLoggingContext(self, instance, methods)

    def _log_method(
        self,
        instance: Any,
        method_name: str,
        metric: Callable[[Any, Any], Any],
    ):
        for attr in method_name.split(".")[:-1]:
            instance = getattr(instance, attr)
        full_name = method_name
        method_name = method_name.split(".")[-1]
        original_method = getattr(instance, method_name)
        assert full_name not in self._logged_methods
        self._logged_methods[full_name] = (
            instance,
            method_name,
            original_method,
        )

        def method_with_hook(*args, **kwargs):
            output = original_method(*args, **kwargs)
            input = args_to_kwargs(original_method, args, kwargs)

            if self._context:
                stats = metric(input=input, output=output)
                if not (stats is None):
                    self.log(name=full_name, data=stats)

            return output

        setattr(instance, method_name, method_with_hook)

    def is_logged(self, method_name: str) -> bool:
        return method_name in self._logged_methods

    @abstractmethod
    def _log(self, data: Any, name: str, context: Dict[str, Any]):
        raise NotImplementedError()

    def log(self, name: str, data: Any):
        self._log(data, name=name, context=self._context)

    @abstractmethod
    def _reset_log(self):
        raise NotImplementedError()

    def reset_log(self):
        self._reset_log()
        self._context = {}

    def detach(self, method_name):
        instance, method_original_name, original_method = self._logged_methods[
            method_name
        ]
        setattr(instance, method_original_name, original_method)
        del self._logged_methods[method_name]

    def detach_all(self):
        logged_methods = list(self._logged_methods)
        for method_name in logged_methods:
            self.detach(method_name)

    def clear(self):
        self.detach_all()
        self.reset_log()

    def get_log(self) -> Optional[Any]:
        return None

    def __del__(self):
        self.clear()


class DummyLogger(Logger):
    def _log(self, data: Any, name: str, context: Dict[str, Any]):
        pass

    def _reset_log(self):
        pass
