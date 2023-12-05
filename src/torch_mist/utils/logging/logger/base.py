from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, ContextManager, Union, List

SPLITS = ["train", "valid", "test"]


class LoggingContext:
    def __init__(self, logger: Any, context: str):
        self.logger = logger
        self.context = context

    def __enter__(self) -> "LoggingContext":
        assert self.logger._context is None
        self.logger._context = self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger._context = None


class Logger:
    def __init__(self):
        self.iteration = 0
        self.epoch = 0
        self._global_info = {}
        self._logged_methods = set()
        self._context = None

    def train(self) -> ContextManager[LoggingContext]:
        return LoggingContext(self, "train")

    def valid(self) -> ContextManager[LoggingContext]:
        return LoggingContext(self, "valid")

    def test(self) -> ContextManager[LoggingContext]:
        return LoggingContext(self, "test")

    def set_global_info(self, **kwargs):
        self._global_info = kwargs

    def log_method(
        self,
        instance: Any,
        method_name: str,
        agg: Optional[
            Callable[[Any], Union[List[Dict[str, Any]], Dict[str, Any]]]
        ] = None,
    ):
        assert method_name not in self._logged_methods
        self._logged_methods.add(method_name)

        original_method = getattr(instance, method_name)

        if agg is None:

            def mean(x):
                return {"value": x.mean().item(), "metric": "mean"}

            agg = mean

        def method_with_hook(*args, **kwargs):
            value = original_method(*args, **kwargs)
            if self._context:
                stats = agg(value.detach())
                if not (stats is None):
                    if isinstance(stats, list):
                        for stat in stats:
                            assert "metric" in stat
                            assert "value" in stat
                            stat["quantity"] = method_name
                            self.log(stat)
                    else:
                        assert isinstance(stats, dict)
                        assert "metric" in stats
                        assert "value" in stats
                        stats["quantity"] = method_name
                        self.log(stats)

            return value

        setattr(instance, method_name, method_with_hook)

    def is_logged(self, method_name: str) -> bool:
        return method_name in self._logged_methods

    @abstractmethod
    def _log(self, data: Dict[str, Any]):
        raise NotImplementedError()

    def log(self, data: Dict[str, Any]):
        data["iteration"] = self.iteration
        data["epoch"] = self.epoch
        data["split"] = self._context
        data.update(self._global_info)
        self._log(data)

    def step(self):
        assert (
            self._context == "train"
        ), "Iterations can be updated only in train context"
        self.iteration += 1

    def new_epoch(self):
        assert (
            self._context == "train"
        ), "Iterations can be updated only in train context"
        self.epoch += 1

    @abstractmethod
    def _reset(self):
        raise NotImplementedError()

    def reset(self):
        self.iteration = 0
        self.epoch = 0
        self._global_info = {}
        self._reset()

    def get_log(self) -> Optional[Any]:
        return None


class DummyLogger(Logger):
    def _log(self, data: Dict[str, Any]):
        pass

    def _reset(self):
        pass
