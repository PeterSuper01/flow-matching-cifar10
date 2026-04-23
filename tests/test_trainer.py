import pytest
import torch
import torch.nn as nn
from trainer import EMA


class Tiny(nn.Module):
    """Single-parameter model for exact numeric verification."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


# ── EMA ───────────────────────────────────────────────────────────────────────

class TestEMA:
    @pytest.fixture
    def tiny(self):
        return Tiny()

    @pytest.fixture
    def ema(self, tiny):
        return EMA(tiny, decay=0.9)

    def test_init_shadow_matches_model(self, tiny, ema):
        for k, v in tiny.state_dict().items():
            assert torch.allclose(ema.shadow[k], v.float())

    def test_update_moves_shadow(self, tiny, ema):
        initial = {k: v.clone() for k, v in ema.shadow.items()}
        for p in tiny.parameters():
            p.data.fill_(1.0)
        ema.update(tiny)
        assert any(not torch.equal(ema.shadow[k], initial[k]) for k in ema.shadow)

    def test_update_decay_math(self, tiny, ema):
        # shadow starts at 0; model weight set to 1; decay=0.9
        # expected: 0 * 0.9 + 1 * 0.1 = 0.1
        tiny.w.data.fill_(1.0)
        ema.update(tiny)
        assert torch.isclose(ema.shadow["w"], torch.tensor([0.1]))

    def test_apply_to_loads_shadow_into_model(self, tiny, ema):
        tiny.w.data.fill_(99.0)   # corrupt the model
        ema.apply_to(tiny)
        assert torch.allclose(tiny.w.data, ema.shadow["w"].to(tiny.w.dtype))

    def test_state_dict_roundtrip(self, ema):
        state = ema.state_dict()
        new_ema = EMA.__new__(EMA)
        new_ema.decay = ema.decay
        new_ema.load_state_dict(state)
        for k in state:
            assert torch.equal(new_ema.shadow[k], ema.shadow[k])

    def test_shadow_stored_as_float(self):
        # EMA shadow must be float32 even if model uses float16 for training stability
        model = Tiny().half()
        ema   = EMA(model, decay=0.999)
        assert all(v.dtype == torch.float32 for v in ema.shadow.values())

    def test_multiple_updates_converge_to_model(self):
        # after many updates with decay close to 0, shadow should equal model weights
        model = Tiny()
        ema   = EMA(model, decay=0.0)   # no smoothing: shadow = model at each step
        model.w.data.fill_(5.0)
        ema.update(model)
        assert torch.isclose(ema.shadow["w"], torch.tensor([5.0]))
