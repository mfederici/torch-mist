from copy import deepcopy
from typing import Callable, TypeVar, Dict, Any
import inspect

import torch
from torch import nn

T = TypeVar("T")


def _is_cache_valid(arguments: Any, stored_arguments: Any):
    if stored_arguments is None:
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


def delete_cache_hook(self, *args, **kwargs):
    if hasattr(self, "__cache"):
        del self.__cache


def cached_method(method: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args, **kwargs):
        arguments = inspect.getcallargs(method, *args, **kwargs)
        self = arguments.pop("self")

        # Attach __cache to the instance
        if not hasattr(self, "__cache"):
            self.__cache = {}
            self.__cache_stats = {}

            if hasattr(self, "__getstate__"):
                original_getstate = self.__getstate__
            else:
                original_getstate = None

            # Wrap the get_state method to exclude the cache
            def wrapped_getstate():
                # delete_cache_hook(self)
                if original_getstate is None:
                    state = self.__dict__
                else:
                    state = original_getstate()

                if "__getstate__" in state:
                    state["__getstate__"] = original_getstate
                if "__cache" in state:
                    del state["__cache"]
                if "__cache_stats" in state:
                    del state["__cache_stats"]
                return state

            self.__getstate__ = wrapped_getstate

            # For nn.Modules, automatically invalidate cache on backwards()
            if isinstance(self, nn.Module):
                self.register_full_backward_hook(delete_cache_hook)

        # Initialize the stats and empty cache
        if not (method.__name__ in self.__cache):
            self.__cache[method.__name__] = None, None

        if not (method.__name__ in self.__cache_stats):
            self.__cache_stats[method.__name__] = {"hits": 0, "miss": 0}

        stored_arguments, stored_value = self.__cache[method.__name__]
        stats = self.__cache_stats[method.__name__]

        if not _is_cache_valid(arguments, stored_arguments):
            value = method(*args, **kwargs)
            self.__cache[method.__name__] = (arguments, value)
            stats["miss"] += 1
        else:
            stats["hits"] += 1
            value = stored_value

        return value

    wrapper.__signature__ = inspect.signature(method)

    return wrapper


def cached_function(method: Callable[..., T]) -> Callable[..., T]:
    stored_arguments = {}
    stored_value = None
    stats = {"hits": 0, "miss": 0}

    def cache_info():
        nonlocal stats
        return stats

    def delete_cache():
        nonlocal stored_value, stored_arguments
        stored_value = None
        stored_arguments = {}

    def wrapper(*args, **kwargs):
        arguments = inspect.getcallargs(method, *args, **kwargs)

        # static cache for functions (no methods)
        nonlocal stats, stored_arguments, stored_value

        if not _is_cache_valid(arguments, stored_arguments):
            value = method(*args, **kwargs)
            stored_value = value
            stored_arguments = arguments
            stats["miss"] += 1
        else:
            stats["hits"] += 1
            value = stored_value

        return value

    wrapper.cache_info = cache_info
    wrapper.delete_cache = delete_cache
    wrapper.__signature__ = inspect.signature(method)
    return wrapper
