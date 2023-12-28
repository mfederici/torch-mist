from __future__ import annotations

from types import TracebackType
from typing import Any, Callable, Optional, List, Tuple, Dict

from torch_mist.utils.logging.metrics import compute_mean


class LoggingContext:
    def __init__(
        self,
        logger: Any,
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
        logger: Any,
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
        for method in self.methods:
            if isinstance(method, tuple):
                method_name, metric = method
            else:
                method_name = method
                metric = compute_mean
            self.logger._log_method(self.instance, method_name, metric)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        for method in self.methods:
            if isinstance(method, tuple):
                method_name, _ = method
            else:
                method_name = method
            self.logger.detach(method_name)
