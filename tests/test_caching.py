import torch

from torch_mist.estimators import js


def test_caching():
    estimator = js(x_dim=1, y_dim=1, hidden_dims=[32])

    x = torch.zeros(10, 1)
    y = torch.zeros(10, 1)

    assert not hasattr(estimator, "__cache")
    estimator.loss(x, y)

    assert hasattr(estimator, "__cache")
    assert "critic_on_negatives" in estimator.__cache
    inputs, output = estimator.__cache["critic_on_negatives"]

    assert torch.equal(inputs["x"], x) and torch.equal(inputs["y"], y)

    # Cache hit
    value = estimator(x, y)

    value_2 = estimator(x, y)
    assert value == value_2

    assert estimator.__cache["critic_on_negatives"] is None

    estimator.loss(x + 1, y)

    # Cache miss
    value = estimator(x, y)

    value_2 = estimator(x, y)
    assert value == value_2

    # Caching disabled
    estimator.__caching_enabled = False
    estimator.loss(x + 1, y)

    assert estimator.__cache["critic_on_negatives"] is None
