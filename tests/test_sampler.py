import pytest
import torch
from models.unet import NUM_CLASSES
from sampler import generate, show_images, CIFAR10_CLASSES


# ── generate ──────────────────────────────────────────────────────────────────

class TestGenerate:
    @pytest.fixture(scope="class")
    def model(self, unet):
        return unet

    def test_output_shape(self, unet):
        y = torch.randint(0, NUM_CLASSES, (8,))
        with torch.no_grad():
            out = generate(unet, y, steps=2)
        assert out.shape == (8, 3, 32, 32)

    def test_output_clamped_to_minus1_1(self, unet):
        y = torch.randint(0, NUM_CLASSES, (4,))
        with torch.no_grad():
            out = generate(unet, y, steps=2)
        assert out.min() >= -1.0
        assert out.max() <=  1.0

    def test_output_finite(self, unet):
        y = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            out = generate(unet, y, steps=2)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("guidance_scale", [0.0, 1.0, 3.0, 7.0])
    def test_various_guidance_scales(self, unet, guidance_scale):
        y = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            out = generate(unet, y, steps=2, guidance_scale=guidance_scale)
        assert out.shape == (4, 3, 32, 32)
        assert torch.isfinite(out).all()

    def test_guidance_scale_affects_output(self, unet):
        # different guidance scales should produce different images
        torch.manual_seed(0)
        y = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            out_low  = generate(unet, y, steps=2, guidance_scale=1.0)
            out_high = generate(unet, y, steps=2, guidance_scale=7.0)
        assert not torch.allclose(out_low, out_high)

    def test_single_sample(self, unet):
        y = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            out = generate(unet, y, steps=2)
        assert out.shape == (1, 3, 32, 32)

    def test_all_classes(self, unet):
        y = torch.arange(NUM_CLASSES, dtype=torch.long)
        with torch.no_grad():
            out = generate(unet, y, steps=2)
        assert out.shape == (NUM_CLASSES, 3, 32, 32)


# ── CIFAR10_CLASSES ───────────────────────────────────────────────────────────

class TestCifar10Classes:
    def test_length_matches_num_classes(self):
        assert len(CIFAR10_CLASSES) == NUM_CLASSES

    def test_all_unique(self):
        assert len(set(CIFAR10_CLASSES)) == len(CIFAR10_CLASSES)

    def test_known_entries(self):
        for name in ("airplane", "dog", "ship"):
            assert name in CIFAR10_CLASSES


# ── show_images ───────────────────────────────────────────────────────────────

class TestShowImages:
    def test_runs_without_error(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")  # headless
        imgs = torch.randn(16, 3, 32, 32)
        fname = str(tmp_path / "out.png")
        show_images(imgs, nrow=4, title="test", fname=fname)
        assert (tmp_path / "out.png").exists()
