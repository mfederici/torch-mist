import numpy as np
import torch

from torch_mist.data.multimixture import MultivariateCorrelatedNormalMixture
from torch_mist.utils.data import SameAttributeSampler
from torch_mist.utils.data import (
    SameAttributeDataLoader,
    SampleDataset,
    sample_same_attributes,
)
from torch.utils.data import DataLoader


def _test_same_attributes_in_batch(dataloader):
    neg_samples = dataloader.batch_sampler.neg_samples
    for batch in dataloader:
        batch_a = _compute_attributes(batch["y"])
        batch_a = batch_a.reshape(neg_samples, -1)
        assert (
            batch_a == batch_a[0].unsqueeze(0)
        ).sum() == batch_a.numel(), "The batch should have the same attributes"


def _compute_attributes(data):
    n_dim = data.shape[-1]
    return ((data > 0).long() * 2 ** torch.arange(n_dim).reshape(1, -1)).sum(
        -1
    )


def test_sampler():
    data = np.arange(120)
    neg_samples = 6
    batch_size = 21

    def f(x):
        return ((x > 30) + 0) + (x > 61)

    dataloader = DataLoader(
        data.reshape(-1, 1),
        batch_sampler=SameAttributeSampler(
            batch_size=batch_size, neg_samples=neg_samples, attributes=f(data)
        ),
    )

    batches = []

    for batch in dataloader:
        batches.append(batch)

    sampled = set()
    for batch in batches:
        batch = batch.reshape(neg_samples + 1, -1, 1)
        b0 = batch[0]
        assert torch.equal(
            f(b0.unsqueeze(0)).repeat(neg_samples + 1, 1, 1), f(batch)
        )
        for e in batch.reshape(-1):
            assert not (e in sampled)
            sampled.add(e)

    assert len(batches) == len(dataloader)


def test_same_attribute_samples():
    n_dim = 2
    p_xy = MultivariateCorrelatedNormalMixture(n_dim=n_dim)
    samples = p_xy.sample([1000])
    a = _compute_attributes(samples["y"])
    dataset = SampleDataset(samples)

    failed = False
    try:
        dataloader = SameAttributeDataLoader(
            dataset, attributes=a[:-1], batch_size=100, neg_samples=9
        )
    except Exception as e:
        print(e)
        failed = True

    assert failed, "The test should fail with incompatible shapes"

    dataloader = SameAttributeDataLoader(
        dataset, attributes=a, batch_size=100, neg_samples=9
    )

    _test_same_attributes_in_batch(dataloader)

    dataloader = DataLoader(dataset, batch_size=100)

    _test_same_attributes_in_batch(
        sample_same_attributes(dataloader, a, neg_samples=9)
    )
