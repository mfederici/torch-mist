from typing import Callable, TypeVar

import torch
T = TypeVar("T")


def cached(method: Callable[..., T]) -> Callable[..., T]:
    def _is_cache_valid(self, key: str, **kwargs):
        assert key in self._cache, f"Key {key} not in cache"
        if self._cache[key] is None:
            return False
        stored_kwargs, stored_value = self._cache[key]
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

    def _reset_cache(self):
        self._cache = {key: None for key in self._cache.keys()}

    def cached_method(self, *args, **kwargs):
        assert len(args) == 0, f"Cached method {method.__name__} accept only named parameters"

        if not hasattr(self, '_cache'):
            self._cache = {}
            self._caching_enabled = True
            self._reset_cache = _reset_cache.__get__(self)

        if not self._caching_enabled:
            return method(*args, **kwargs)

        key = method.__name__
        if key not in self._cache or not _is_cache_valid(self, key, **kwargs):
            value = method(self, **kwargs)
            self._cache[key] = (kwargs, value)
        else:
            value = self._cache[key][1]
        return value

    return cached_method


def reset_cache_after_call(method: Callable[..., T]) -> Callable[..., T]:
    def reset_cache(self, *args, **kwargs):
        value = method(self, *args, **kwargs)
        if hasattr(self, '_reset_cache'):
            self._reset_cache()
        return value

    return reset_cache


def reset_cache_before_call(method: Callable[..., T]) -> Callable[..., T]:
    def reset_cache(self, *args, **kwargs):
        if hasattr(self, '_reset_cache'):
            self._reset_cache()
        return method(self, *args, **kwargs)

    return reset_cache
