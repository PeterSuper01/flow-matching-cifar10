import pytest
import torch
from torch.utils.data import TensorDataset
from data import get_loader


def _fake_dataset(n=100):
    imgs   = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    return TensorDataset(imgs, labels)


class TestGetLoader:
    def test_batch_shape(self):
        loader = get_loader(_fake_dataset(), batch_size=16, num_workers=0)
        imgs, labels = next(iter(loader))
        assert imgs.shape   == (16, 3, 32, 32)
        assert labels.shape == (16,)

    def test_loader_length(self):
        loader = get_loader(_fake_dataset(n=100), batch_size=10, num_workers=0)
        assert len(loader) == 10

    def test_shuffle_changes_order(self):
        dataset = _fake_dataset(n=50)
        loader  = get_loader(dataset, batch_size=50, num_workers=0)
        batch_a = next(iter(loader))[0].clone()
        batch_b = next(iter(loader))[0].clone()
        # With shuffle=True two consecutive full-batch iterations should differ
        # (probability of identical order is 1/50! ≈ 0)
        assert not torch.equal(batch_a, batch_b)
