import numpy as np
import torch

from torch_mist.data.multimixture import MultivariateCorrelatedNormalMixture
from torch_mist.data.multivariate import JointMultivariateNormal
from torch_mist.utils.data import (
    DistributionDataLoader,
    SameAttributeDataLoader,
    SampleDataset,
    sample_same_attributes,
)
from torch.utils.data import DataLoader


def _test_same_attributes_in_batch(dataloader):
    for batch in dataloader:
        n_dim = batch["x"].shape[-1]
        batch_a = (
            (batch["x"] > 0).long() * 2 ** torch.arange(n_dim).reshape(1, -1)
        ).sum(-1)
        assert (batch_a == batch_a[0]).sum() == batch_a.shape[
            0
        ], "The batch should have the same attributes"


def test_dataloaders():
    p_xy = MultivariateCorrelatedNormalMixture(n_dim=5)
    dataloader = DistributionDataLoader(p_xy, batch_size=100, max_samples=1000)
    count = 0
    for batch in dataloader:
        count += 1
        assert "x" in batch and "y" in batch

    assert count == 10, "There should be 10 batches"


def test_same_attribute_samples():
    n_dim = 2
    p_xy = MultivariateCorrelatedNormalMixture(n_dim=n_dim)
    samples = p_xy.sample([1000])
    a = (
        (samples["x"] > 0).long() * 2 ** torch.arange(n_dim).reshape(1, -1)
    ).sum(-1)
    dataset = SampleDataset(samples)

    failed = False
    try:
        dataloader = SameAttributeDataLoader(
            dataset, attributes=a[:-1], batch_size=100
        )
    except Exception as e:
        print(e)
        failed = True

    assert failed, "The test should fail with incompatible shapes"

    dataloader = SameAttributeDataLoader(dataset, attributes=a, batch_size=100)

    _test_same_attributes_in_batch(dataloader)

    dataloader = DataLoader(dataset, batch_size=100)

    _test_same_attributes_in_batch(sample_same_attributes(dataloader, a))


def test_joint_dist():
    p_XY = JointMultivariateNormal(n_dim=3)
    p_X = p_XY.marginal("x")
    h_x = p_XY.entropy("x")
    h_x_ = p_X.entropy()
    p_Y_X = p_XY.conditional("x")
    samples = p_XY.sample([100000])
    log_p_xy = p_XY.log_prob(**samples).mean()
    log_p_x = p_X.log_prob(samples["x"]).mean()
    log_p_y_x = p_Y_X.condition(samples["x"]).log_prob(samples["y"]).mean()

    assert np.isclose(log_p_xy, log_p_x + log_p_y_x, atol=1e-3)

    assert np.isclose(h_x, h_x_, atol=1e-3)

    assert np.isclose(h_x, -log_p_x, atol=1e-2)

    p_XY = MultivariateCorrelatedNormalMixture(n_dim=3)
    p_X = p_XY.marginal("x")
    h_x = p_XY.entropy("x")
    p_Y_X = p_XY.conditional("x")
    samples = p_XY.sample([100000])
    log_p_xy = p_XY.log_prob(**samples).mean()
    log_p_x = p_X.log_prob(samples["x"]).mean()
    log_p_y_x = p_Y_X.condition(samples["x"]).log_prob(samples["y"]).mean()

    assert np.isclose(log_p_xy, log_p_x + log_p_y_x, atol=1e-3)

    assert np.isclose(h_x, -log_p_x, atol=1e-2)
