import numpy as np
import torch
from torch.utils.data import DataLoader

from torch_mist.data.multimixture import MultivariateCorrelatedNormalMixture
from torch_mist.data.multivariate import JointMultivariateNormal
from torch_mist.utils.data.dataset import DistributionDataset


def test_dataloaders():
    p_xy = MultivariateCorrelatedNormalMixture(n_dim=5)
    dataset = DistributionDataset(p_xy, max_samples=1000)
    count = 0
    dataloader = DataLoader(dataset, batch_size=100)
    for batch in dataloader:
        count += 1
        assert "x" in batch and "y" in batch

    assert count == 10, "There should be 10 batches"


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

    print("joint", p_XY)
    print("conditional", p_Y_X)
    print("marginal", p_X)

    assert np.isclose(log_p_xy, log_p_x + log_p_y_x, atol=1e-3)

    assert np.isclose(h_x, h_x_, atol=1e-3)

    assert np.isclose(h_x, -log_p_x, atol=1e-2)

    # Make sure we get an error if we condition on variables on which the support is not defined
    failed = False
    try:
        p_XY.conditional("v")
    except ValueError as v:
        print(v)
        failed = True
    assert failed

    failed = False
    try:
        p_XY.condition(v=torch.zeros(10, 3))
    except ValueError as v:
        print(v)
        failed = True
    assert failed

    p_XY = MultivariateCorrelatedNormalMixture(n_dim=3)
    p_X = p_XY.marginal("x")
    h_x = p_XY.entropy("x")
    p_Y_X = p_XY.conditional("x")
    samples = p_XY.sample([100000])
    log_p_xy = p_XY.log_prob(**samples).mean().item()
    log_p_x = p_X.log_prob(samples["x"]).mean().item()
    log_p_y_x = (
        p_Y_X.condition(samples["x"]).log_prob(samples["y"]).mean().item()
    )

    assert np.isclose(log_p_xy, log_p_x + log_p_y_x, atol=1e-3)

    assert np.isclose(h_x, -log_p_x, atol=1e-2)
