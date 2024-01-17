import torch

from torch_mist.estimators import js
from torch_mist.utils.caching import cached_function


def test_caching():
    estimator = js(x_dim=1, y_dim=1, hidden_dims=[32])

    x = torch.zeros(10, 1)
    y = torch.zeros(10, 1)
    assert not hasattr(estimator, "__cache")

    value_1 = estimator(x, y)
    assert hasattr(estimator, "__cache_stats")
    cache_stats = estimator.__cache_stats["unnormalized_log_ratio"]
    assert cache_stats["miss"] == 1 and cache_stats["hits"] == 0

    # Cache hit
    value_2 = estimator(x, y)
    cache_stats = estimator.__cache_stats["unnormalized_log_ratio"]
    assert cache_stats["miss"] == 1 and cache_stats["hits"] == 1

    # Cache miss
    value_3 = estimator(x + 1, y)
    cache_stats = estimator.__cache_stats["unnormalized_log_ratio"]
    assert cache_stats["miss"] == 2 and cache_stats["hits"] == 1

    # Check the values
    assert value_1 == value_2 and value_1 != value_3

    # Make sure the cache is automatically invalidated when we use backwards()
    assert len(estimator.__cache) > 0

    estimator(x + 1, y).backward()

    assert len(estimator.__cache) == 0


@cached_function
def f(x):
    pass


def test_function_cache():
    a = torch.LongTensor([1])
    stats = f.cache_info()
    assert stats["hits"] == 0 and stats["miss"] == 0
    f(a)
    stats = f.cache_info()
    assert stats["hits"] == 0 and stats["miss"] == 1

    f(a)
    stats = f.cache_info()
    assert stats["hits"] == 1 and stats["miss"] == 1

    f.delete_cache()
    f(a)
    stats = f.cache_info()
    assert stats["hits"] == 1 and stats["miss"] == 2
