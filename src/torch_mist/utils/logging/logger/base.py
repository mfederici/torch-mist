from __future__ import annotations

from abc import abstractmethod
from types import TracebackType
from typing import Any, Callable, Optional, Union, List, Tuple, Dict

from torch_mist.utils.misc import args_to_kwargs

SPLITS = ["train", "valid", "test"]


class LoggingContext:
    def __init__(
        self,
        logger: "Logger",
        context_name: str,
        context_value: Any,
        dependencies: Optional[Dict[str, Any]] = None,
        persistent: bool = False,
    ):
        self.logger = logger
        self.context_name = context_name
        self.context_value = context_value
        if dependencies is None:
            dependencies = {}
        self.dependencies = dependencies
        self.persistent = persistent

    def _validate(self):
        if not self.persistent:
            assert not (self.context_name in self.logger._context)
        for name, value in self.dependencies.items():
            assert name in self.logger._context
            assert self.logger._context[name] == value

    def __enter__(self):
        self._validate()
        self.logger._context[self.context_name] = self.context_value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        if not self.persistent:
            del self.logger._context[self.context_name]


class IncrementalLoggingContext(LoggingContext):
    def __init__(
        self,
        logger: "Logger",
        context_name: str,
        dependencies: Optional[Dict[str, Any]] = None,
        initial_value: int = 0,
    ):
        super().__init__(
            logger=logger,
            context_name=context_name,
            context_value=initial_value,
            dependencies=dependencies,
            persistent=True,
        )

    def __enter__(self):
        self._validate()
        if not (self.context_name in self.logger._context):
            self.logger._context[self.context_name] = self.context_value
        else:
            self.logger._context[self.context_name] += 1


class MethodsLoggingContext:
    def __init__(
        self,
        logger: Any,
        instance: Any,
        methods: List[Tuple[str, Callable[[Any, Any], Any]]],
    ):
        self.logger = logger
        self.instance = instance
        self.methods = methods

    def __enter__(self):
        for method_name, metric in self.methods:
            self.logger._log_method(self.instance, method_name, metric)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        for method_name, _ in self.methods:
            self.logger.detach(method_name)


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
        method_name = method_name.split(".")[-1]
        original_method = getattr(instance, method_name)
        assert method_name not in self._logged_methods
        self._logged_methods[method_name] = (instance, original_method)

        def method_with_hook(*args, **kwargs):
            output = original_method(*args, **kwargs)
            input = args_to_kwargs(original_method, args, kwargs)

            if self._context:
                stats = metric(input=input, output=output)
                if not (stats is None):
                    self.log(name=method_name, data=stats)

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
        instance, original_method = self._logged_methods[method_name]
        setattr(instance, method_name, original_method)
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
