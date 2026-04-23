import pytest
import torch
import torch.nn as nn
from models.unet import UNet, ResBlock, SinusoidalPosEmb, NUM_CLASSES


# ── SinusoidalPosEmb ──────────────────────────────────────────────────────────

class TestSinusoidalPosEmb:
    def test_output_shape(self):
        emb = SinusoidalPosEmb(dim=64)
        out = emb(torch.rand(8))
        assert out.shape == (8, 64)

    def test_output_finite(self):
        emb = SinusoidalPosEmb(dim=256)
        out = emb(torch.rand(16))
        assert torch.isfinite(out).all()

    def test_boundary_values(self):
        # t=0 and t=1 are common edge cases in flow matching
        emb = SinusoidalPosEmb(dim=64)
        out = emb(torch.tensor([0.0, 1.0]))
        assert torch.isfinite(out).all()

    def test_different_inputs_different_outputs(self):
        emb = SinusoidalPosEmb(dim=64)
        out_a = emb(torch.zeros(1))
        out_b = emb(torch.ones(1))
        assert not torch.allclose(out_a, out_b)


# ── ResBlock ──────────────────────────────────────────────────────────────────

class TestResBlock:
    @pytest.mark.parametrize("in_ch,out_ch,H,W", [
        (64,  64, 32, 32),   # same channels — encoder level 0
        (64, 128, 16, 16),   # channel expansion — encoder level 1
        (384, 128, 16, 16),  # cat-skip expansion — decoder level 2
        (192,  64, 32, 32),  # cat-skip expansion — decoder level 1
    ])
    def test_output_shape(self, in_ch, out_ch, H, W):
        block = ResBlock(in_ch, out_ch, time_dim=256)
        x     = torch.randn(4, in_ch, H, W)
        t_emb = torch.randn(4, 256)
        out   = block(x, t_emb)
        assert out.shape == (4, out_ch, H, W)

    def test_skip_is_identity_when_same_channels(self):
        block = ResBlock(64, 64, time_dim=256)
        assert isinstance(block.skip, nn.Identity)

    def test_skip_is_conv_when_channels_differ(self):
        block = ResBlock(64, 128, time_dim=256)
        assert isinstance(block.skip, nn.Conv2d)
        assert block.skip.kernel_size == (1, 1)

    def test_spatial_size_preserved(self):
        block = ResBlock(32, 64, time_dim=128)
        x     = torch.randn(2, 32, 17, 17)   # odd spatial size
        t_emb = torch.randn(2, 128)
        out   = block(x, t_emb)
        assert out.shape[-2:] == (17, 17)


# ── UNet ──────────────────────────────────────────────────────────────────────

class TestUNet:
    def test_output_shape(self, unet, t, x, y):
        with torch.no_grad():
            out = unet(t, x, y)
        assert out.shape == x.shape  # velocity field matches input shape

    @pytest.mark.parametrize("B", [1, 4, 8])
    def test_various_batch_sizes(self, B):
        model = UNet().eval()
        t = torch.rand(B)
        x = torch.randn(B, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (B,))
        with torch.no_grad():
            out = model(t, x, y)
        assert out.shape == (B, 3, 32, 32)

    def test_null_token(self, unet, t, x):
        # index NUM_CLASSES is the unconditional null token used in CFG dropout
        y_null = torch.full((len(t),), NUM_CLASSES, dtype=torch.long)
        with torch.no_grad():
            out = unet(t, x, y_null)
        assert out.shape == x.shape

    def test_output_finite(self, unet, t, x, y):
        with torch.no_grad():
            out = unet(t, x, y)
        assert torch.isfinite(out).all()

    def test_conditional_and_unconditional_differ(self, unet, t, x):
        y_cond = torch.zeros(len(t), dtype=torch.long)
        y_null = torch.full((len(t),), NUM_CLASSES, dtype=torch.long)
        with torch.no_grad():
            v_cond   = unet(t, x, y_cond)
            v_uncond = unet(t, x, y_null)
        assert not torch.allclose(v_cond, v_uncond)

    def test_parameter_count(self):
        n = sum(p.numel() for p in UNet().parameters())
        assert 5_000_000 < n < 20_000_000  # default config is ~8.9M

    def test_all_classes_accepted(self, unet, t, x):
        for cls in range(NUM_CLASSES):
            y = torch.full((len(t),), cls, dtype=torch.long)
            with torch.no_grad():
                out = unet(t, x, y)
            assert out.shape == x.shape
