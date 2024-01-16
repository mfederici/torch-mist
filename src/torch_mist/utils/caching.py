from typing import Callable, TypeVar, Dict, Any
import inspect

import torch

T = TypeVar("T")


def cached(method: Callable[..., T]) -> Callable[..., T]:
    stored_arguments = {}
    stored_value = None
    miss = 0
    hits = 0

    def _is_cache_valid(arguments: Any, stored_arguments: Any):
        if stored_value is None:
            return False

        if torch.is_tensor(arguments):
            if not torch.equal(stored_arguments, arguments):
                return False
        elif isinstance(arguments, dict):
            if stored_arguments.keys() != arguments.keys():
                return False
            for k in arguments:
                if not _is_cache_valid(arguments[k], stored_arguments[k]):
                    return False
        elif isinstance(arguments, list) or isinstance(arguments, tuple):
            if len(arguments) != len(stored_arguments):
                return False
            for i in range(len(arguments)):
                if not _is_cache_valid(arguments[i], stored_arguments[i]):
                    return False
        else:
            if stored_arguments != arguments:
                return False
        return True

    def wrapper(*args, **kwargs):
        nonlocal hits, miss, stored_arguments, stored_value

        arguments = inspect.getcallargs(method, *args, **kwargs)
        if "self" in arguments:
            arguments.pop("self")

        if not _is_cache_valid(arguments, stored_arguments):
            value = method(*args, **kwargs)
            stored_value = value
            stored_arguments = arguments
            miss += 1
        else:
            hits += 1
            value = stored_value
        return value

    def cache_info():
        nonlocal hits, miss
        return {"hits": hits, "miss": miss}

    wrapper.cache_info = cache_info
    wrapper.__signature__ = inspect.signature(method)
    return wrapper
