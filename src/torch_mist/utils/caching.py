from typing import Callable, TypeVar
import inspect

import torch

T = TypeVar("T")


def cached(method: Callable[..., T]) -> Callable[..., T]:
    def _is_cache_valid(self, key: str, **kwargs):
        assert key in self.__cache, f"Key {key} not in cache"
        if self.__cache[key] is None:
            return False
        stored_kwargs, stored_value = self.__cache[key]
        if stored_kwargs.keys() != kwargs.keys():
            return False

        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if not torch.equal(stored_kwargs[k], v):
                    return False
            else:
                if stored_kwargs[k] != v:
                    return False

        return True

    def cached_method(self, *args, **kwargs):
        # transform args in kwargs
        keys = inspect.signature(method).parameters.keys()
        unused_keys = [key for key in keys if key not in kwargs]
        if "self" in unused_keys:
            unused_keys.remove("self")
        new_kwargs = {unused_keys[i]: arg for i, arg in enumerate(args)}
        kwargs = {**kwargs, **new_kwargs}

        if not hasattr(self, "__cache"):
            self.__cache = {}
            self.__caching_enabled = True

        if not self.__caching_enabled:
            return method(self, *args, **kwargs)

        key = method.__name__
        if key not in self.__cache or not _is_cache_valid(self, key, **kwargs):
            value = method(self, **kwargs)
            # print(f"Cache miss for {key}")
            self.__cache[key] = (kwargs, value)
        else:
            # print(f"Cache hit for {key}")
            value = self.__cache[key][1]
        return value

    return cached_method


def reset_cache_after_call(method: Callable[..., T]) -> Callable[..., T]:
    def reset_cache(self, *args, **kwargs):
        value = method(self, *args, **kwargs)
        if hasattr(self, "__cache"):
            self.__cache = {key: None for key in self.__cache.keys()}
        return value

    return reset_cache


def reset_cache_before_call(method: Callable[..., T]) -> Callable[..., T]:
    def reset_cache(self, *args, **kwargs):
        if hasattr(self, "__cache"):
            self.__cache = {key: None for key in self.__cache.keys()}
        return method(self, *args, **kwargs)

    return reset_cache
