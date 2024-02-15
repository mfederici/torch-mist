from __future__ import annotations

import inspect
import os
from abc import abstractmethod
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from typing import Any, Callable, Optional, Union, List, Tuple, Dict

from torch_mist.utils.logging.metrics import compute_mean


class Logger:
    def __init__(self, log_dir: str = ".", log_every: int = 1):
        self.log_dir = log_dir
        self.log_every = log_every
        self._buffer = {}
        self._logged_methods = {}
        self._split = None
        self._iteration = 0
        self._epoch = 0

    @contextmanager
    def train(self):
        self.on_split_start("train")
        try:
            yield
        finally:
            self.on_split_end("train")

    @contextmanager
    def valid(self):
        self.on_split_start("valid")
        try:
            yield
        finally:
            self.on_split_end("valid")

    @contextmanager
    def test(self):
        self.on_split_start("test")
        try:
            yield
        finally:
            self.on_split_end("test")

    @contextmanager
    def iteration(self):
        self.on_iteration_start()
        try:
            yield
        finally:
            self.on_iteration_end()

    @contextmanager
    def epoch(self):
        self.on_epoch_start()
        try:
            yield
        finally:
            self.on_epoch_end()

    def on_split_start(self, name: str):
        assert self._split is None
        self._split = name

    def on_split_end(self, name: str):
        if name != "train":
            self._log_buffer(force_logging=name)
        self._split = None

    def on_iteration_start(self):
        pass

    def on_iteration_end(self):
        self._iteration += 1
        self._log_buffer()
        self._log_buffer()

    def on_epoch_start(self):
        self._epoch += 1

    def on_epoch_end(self):
        pass

    @contextmanager
    def logged_methods(
        self,
        instance: Any,
        methods: List[Union[str, Tuple[str, Callable[[Any, Any], Any]]]],
    ):
        for method in methods:
            if isinstance(method, tuple):
                method_name, metric = method
            else:
                method_name = method
                metric = compute_mean
            self.add_logging_hook(instance, method_name, metric)
        try:
            yield
        finally:
            for method in methods:
                if isinstance(method, tuple):
                    method_name, _ = method
                else:
                    method_name = method
                self.detach_hook(method_name)

    def add_logging_hook(
        self,
        instance: Any,
        method_name: str,
        metric: Callable[[Any, Any], Dict[str, Any]],
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
            arguments = inspect.getcallargs(original_method, *args, **kwargs)

            if self._split:
                stats = metric(input=arguments, output=output)
                if not (stats is None):
                    if not isinstance(stats, dict):
                        stats = {"": stats}
                    self.log(name=full_name, data_dict=stats)

            return output

        method_with_hook.__signature__ = inspect.signature(original_method)

        setattr(instance, method_name, method_with_hook)

    def detach_hook(self, method_name):
        instance, method_original_name, original_method = self._logged_methods[
            method_name
        ]
        setattr(instance, method_original_name, original_method)
        del self._logged_methods[method_name]

    def is_logged(self, method_name: str) -> bool:
        return method_name in self._logged_methods

    def _log_buffer(self, force_logging: Optional[str] = None):
        keys_to_delete = []
        for full_name, (orig_iteration, data) in self._buffer.items():
            split = full_name.split("/")[0]
            name = "/".join(full_name.split("/")[1:])
            if (
                self._iteration - orig_iteration >= self.log_every
                or force_logging == split
            ):
                agg_values = {}
                for metric, values in data.items():
                    first_value = values[0]
                    if (
                        isinstance(first_value, float)
                        or (
                            isinstance(first_value, torch.Tensor)
                            and first_value.numel() == 1
                        )
                        or (
                            isinstance(first_value, np.ndarray)
                            and first_value.size == 1
                        )
                    ):
                        agg_values[metric] = np.mean(values)
                    else:
                        agg_values[metric] = first_value

                # Remove the dictionary wrap if necessary
                if len(agg_values) == 1:
                    if "" in agg_values:
                        agg_values = agg_values[""]

                self._log(
                    data=agg_values,
                    name=name,
                    iteration=self._iteration,
                    epoch=self._epoch,
                    split=split,
                )
                keys_to_delete.append(full_name)

        for full_name in keys_to_delete:
            del self._buffer[full_name]

    def log(self, name: str, data_dict: Any):
        name = f"{self._split}/{name}"
        if not (name in self._buffer):
            self._buffer[name] = (
                self._iteration,
                {metric: [value] for metric, value in data_dict.items()},
            )
        else:
            for metric, value in data_dict.items():
                self._buffer[name][1][metric].append(value)

    @abstractmethod
    def _log(
        self, data: Any, name: str, iteration: int, epoch: int, split: str
    ):
        raise NotImplementedError()

    @abstractmethod
    def _reset_log(self):
        raise NotImplementedError()

    def reset_log(self):
        self._reset_log()
        self._split = None
        self._iteration = 0
        self._epoch = 0

    def detach_all_hooks(self):
        logged_methods = list(self._logged_methods)
        for method_name in logged_methods:
            self.detach_hook(method_name)

    def clear(self):
        self.detach_all_hooks()
        self.reset_log()

    def __del__(self):
        self.clear()

    def get_log(self) -> Optional[Any]:
        return None

    @abstractmethod
    def save_log(self):
        raise NotImplementedError()

    @abstractmethod
    def save_model(self, model: nn.Module, name: str):
        filepath = os.path.join(self.log_dir, name)
        torch.save(model, filepath)


class DummyLogger(Logger):
    def _log(self, **kwargs):
        pass

    def _reset_log(self):
        pass
