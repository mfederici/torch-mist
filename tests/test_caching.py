import torch

from torch_mist.estimators import js


def test_caching():
    estimator = js(x_dim=1, y_dim=1, hidden_dims=[32])

    x = torch.zeros(10, 1)
    y = torch.zeros(10, 1)

    cache_info = estimator.unnormalized_log_ratio.cache_info()
    assert cache_info["miss"] == 0 and cache_info["hits"] == 0

    value_1 = estimator.loss(x, y)
    assert hasattr(estimator.unnormalized_log_ratio, "cache_info")
    cache_info = estimator.unnormalized_log_ratio.cache_info()
    assert cache_info["miss"] == 1 and cache_info["hits"] == 0

    # Cache hit
    value_2 = estimator.loss(x, y)
    cache_info = estimator.unnormalized_log_ratio.cache_info()
    assert cache_info["miss"] == 1 and cache_info["hits"] == 1

    # Cache miss
    value_3 = estimator.loss(x + 1, y)
    cache_info = estimator.unnormalized_log_ratio.cache_info()
    assert cache_info["miss"] == 2 and cache_info["hits"] == 1

    assert value_1 == value_2 and value_1 != value_3
