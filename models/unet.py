import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        # freqs: (dim/2,)  —  log-spaced frequencies from 1 to 1/10000
        emb = t.float()[:, None] * freqs[None]   # (B, dim/2)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class ResBlock(nn.Module):
    """
    Residual block conditioned on a time (+ class) embedding.

    x      : (B, in_ch,  H, W)
    t_emb  : (B, time_dim)
    return : (B, out_ch, H, W)   — spatial size H×W is preserved
    """
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1     = nn.GroupNorm(8, in_ch)
        self.conv1     = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2     = nn.GroupNorm(8, out_ch)
        self.conv2     = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip      = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))           # (B, out_ch, H, W)
        h = h + self.time_proj(t_emb)[:, :, None, None] # broadcast (B, out_ch, 1, 1) → (B, out_ch, H, W)
        h = self.conv2(F.silu(self.norm2(h)))            # (B, out_ch, H, W)
        return h + self.skip(x)                          # (B, out_ch, H, W)


class UNet(nn.Module):
    """
    Conditioned UNet velocity field for flow matching.

    Default config  base_ch=64, ch_mult=(1,2,4), 32×32 input
    ─────────────────────────────────────────────────────────
    Encoder
      level 0 : (B,   3, 32, 32) → (B,  64, 32, 32)  [ResBlock ×2]
                                 → (B,  64, 16, 16)  [stride-2 conv, saved as skip]
      level 1 : (B,  64, 16, 16) → (B, 128, 16, 16)  [ResBlock ×2]
                                 → (B, 128,  8,  8)  [stride-2 conv, saved as skip]
      level 2 : (B, 128,  8,  8) → (B, 256,  8,  8)  [ResBlock ×2, no downsample]

    Bottleneck  (B, 256,  8,  8) → (B, 256,  8,  8)  [ResBlock ×2]

    Decoder
      level 2 : (B, 256,  8,  8) → (B, 256, 16, 16)  [ConvTranspose2d ×2]
                cat skip (B, 128, 16, 16) → (B, 384, 16, 16)
                                          → (B, 128, 16, 16)  [ResBlock ×2]
      level 1 : (B, 128, 16, 16) → (B, 128, 32, 32)  [ConvTranspose2d ×2]
                cat skip (B,  64, 32, 32) → (B, 192, 32, 32)
                                          → (B,  64, 32, 32)  [ResBlock ×2]

    Output      (B,  64, 32, 32) → (B,   3, 32, 32)  [GroupNorm + SiLU + 1×1 conv]

    Class label y is embedded and added to the time embedding (CFG-style).
    Index NUM_CLASSES acts as the null / unconditional token.
    """
    def __init__(self, in_ch=3, base_ch=64, ch_mult=(1, 2, 4), time_dim=256,
                 num_classes=NUM_CLASSES):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]  # e.g. [64, 128, 256]

        # time_emb: (B,) → (B, time_dim)  via sinusoidal → MLP
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        # class_emb: int index → (B, time_dim);  index num_classes = null token
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)
        self.init_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        # Encoder: two ResBlocks per level + one stride-2 downsample (except last)
        self.enc_blocks  = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, ch in enumerate(chs):
            in_c = chs[i - 1] if i > 0 else chs[0]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock(in_c, ch, time_dim),
                ResBlock(ch,   ch, time_dim),
            ]))
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))

        # Bottleneck: two ResBlocks at the coarsest resolution
        self.mid = nn.ModuleList([
            ResBlock(chs[-1], chs[-1], time_dim),
            ResBlock(chs[-1], chs[-1], time_dim),
        ])

        # Decoder: ConvTranspose2d upsample → cat skip → two ResBlocks
        self.upsamples  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            in_c, out_c = chs[i], chs[i - 1]
            self.upsamples.append(nn.ConvTranspose2d(in_c, in_c, 4, stride=2, padding=1))
            self.dec_blocks.append(nn.ModuleList([
                ResBlock(in_c + out_c, out_c, time_dim),  # in_c from up, out_c from skip
                ResBlock(out_c,        out_c, time_dim),
            ]))

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], in_ch, 1)  # 1×1 projection back to image channels

    def forward(self, t, x, y):
        # ── conditioning ──────────────────────────────────────────────────────
        # t: (B,)   x: (B, 3, 32, 32)   y: (B,) class indices
        t_emb = self.time_emb(t) + self.class_emb(y)  # (B, 256)

        # ── encoder ───────────────────────────────────────────────────────────
        h = self.init_conv(x)   # (B, 3, 32, 32) → (B, 64, 32, 32)

        skips, down_idx = [], 0
        for i, (r1, r2) in enumerate(self.enc_blocks):
            h = r2(r1(h, t_emb), t_emb)         # (B, ch[i], H, W)
            if i < len(self.enc_blocks) - 1:
                skips.append(h)                  # save for skip connection
                h = self.downsamples[down_idx](h)  # H, W → H/2, W/2
                down_idx += 1
        # skips: [(B, 64, 32, 32), (B, 128, 16, 16)]
        # h after encoder: (B, 256, 8, 8)

        # ── bottleneck ────────────────────────────────────────────────────────
        h = self.mid[1](self.mid[0](h, t_emb), t_emb)  # (B, 256, 8, 8)

        # ── decoder ───────────────────────────────────────────────────────────
        for up, (r1, r2) in zip(self.upsamples, self.dec_blocks):
            h = torch.cat([up(h), skips.pop()], dim=1)  # upsample + concat skip → doubled channels
            h = r2(r1(h, t_emb), t_emb)                 # (B, out_c, H, W)

        # ── output ────────────────────────────────────────────────────────────
        return self.out_conv(F.silu(self.out_norm(h)))   # (B, 64, 32, 32) → (B, 3, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# AdaGN variant (Scheme C): same sinusoidal time emb + scale+shift conditioning
# ─────────────────────────────────────────────────────────────────────────────

class AdaGN(nn.Module):
    """
    Adaptive GroupNorm: replaces (GroupNorm + additive time) with (GroupNorm + γ·h + β).
    γ and β are predicted from t_emb via a linear layer zero-initialised so the block
    starts as an identity transform.
    """
    def __init__(self, num_channels, time_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.to_scale_shift = nn.Linear(time_dim, num_channels * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h, t_emb):
        h = self.norm(h)
        scale_shift = self.to_scale_shift(F.silu(t_emb))        # (B, 2C)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlockAdaGN(nn.Module):
    """ResBlock that conditions via AdaGN (scale+shift) instead of additive projection."""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = AdaGN(in_ch,  time_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGN(out_ch, time_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x, t_emb)))
        h = self.conv2(F.silu(self.norm2(h, t_emb)))
        return h + self.skip(x)


class UNetAdaGN(nn.Module):
    """
    UNet with sinusoidal time embedding + AdaGN (scale+shift) conditioning.
    Same architecture and interface as UNet — drop-in replacement for training/inference.
    """
    def __init__(self, in_ch=3, base_ch=64, ch_mult=(1, 2, 4), time_dim=256,
                 num_classes=NUM_CLASSES):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, time_dim)
        self.init_conv  = nn.Conv2d(in_ch, chs[0], 3, padding=1)

        self.enc_blocks  = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, ch in enumerate(chs):
            in_c = chs[i - 1] if i > 0 else chs[0]
            self.enc_blocks.append(nn.ModuleList([
                ResBlockAdaGN(in_c, ch, time_dim),
                ResBlockAdaGN(ch,   ch, time_dim),
            ]))
            if i < len(chs) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))

        self.mid = nn.ModuleList([
            ResBlockAdaGN(chs[-1], chs[-1], time_dim),
            ResBlockAdaGN(chs[-1], chs[-1], time_dim),
        ])

        self.upsamples  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            in_c, out_c = chs[i], chs[i - 1]
            self.upsamples.append(nn.ConvTranspose2d(in_c, in_c, 4, stride=2, padding=1))
            self.dec_blocks.append(nn.ModuleList([
                ResBlockAdaGN(in_c + out_c, out_c, time_dim),
                ResBlockAdaGN(out_c,        out_c, time_dim),
            ]))

        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_conv = nn.Conv2d(chs[0], in_ch, 1)

    def forward(self, t, x, y):
        t_emb = self.time_emb(t) + self.class_emb(y)

        h = self.init_conv(x)

        skips, down_idx = [], 0
        for i, (r1, r2) in enumerate(self.enc_blocks):
            h = r2(r1(h, t_emb), t_emb)
            if i < len(self.enc_blocks) - 1:
                skips.append(h)
                h = self.downsamples[down_idx](h)
                down_idx += 1

        h = self.mid[1](self.mid[0](h, t_emb), t_emb)

        for up, (r1, r2) in zip(self.upsamples, self.dec_blocks):
            h = torch.cat([up(h), skips.pop()], dim=1)
            h = r2(r1(h, t_emb), t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
