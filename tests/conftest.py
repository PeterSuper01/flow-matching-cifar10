import pytest
import torch
from models.unet import UNet, NUM_CLASSES

B = 4  # default batch size for all tests


@pytest.fixture(scope="module")
def unet():
    """Untrained UNet in eval mode, reused across tests in the same module."""
    return UNet().eval()


@pytest.fixture
def t():
    return torch.rand(B)


@pytest.fixture
def x():
    return torch.randn(B, 3, 32, 32)


@pytest.fixture
def y():
    return torch.randint(0, NUM_CLASSES, (B,))
