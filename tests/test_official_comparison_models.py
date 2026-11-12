"""Official-adapter and channel-fusion compatibility contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import io
import pytest
from phc2d.models import build_model, architecture_fingerprint, state_dict_schema_hash


def test_channel_fusion_preserves_initial_output_and_legacy_schema():
    torch = pytest.importorskip("torch")
    args = dict(input_shape=(1, 8, 8), output_shape=(3, 2), width=8, depth=2,
                modes=(2, 2), norm_groups=2, local_kernel=3, drop_path=0.)
    torch.manual_seed(8)
    old = build_model("global_local_band_operator_v1", **args).eval()
    old_keys, old_schema, old_fingerprint = list(old.state_dict()), state_dict_schema_hash(old), architecture_fingerprint(old)
    torch.manual_seed(8)
    new = build_model("global_local_channel_gate_v1", **args).eval()
    assert sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()) == 32
    state = new.state_dict()
    for name, value in old.state_dict().items():
        assert torch.equal(state[name], value)
    x = torch.randn(2, 1, 8, 8)
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)
    new(x).square().mean().backward()
    for block in new.blocks:
        weights = block.fusion.normalized_weights()
        torch.testing.assert_close(weights.sum(dim=0), torch.ones(8))
        grad = block.fusion.channel_logits.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
    assert list(old.state_dict()) == old_keys
    assert state_dict_schema_hash(old) == old_schema and architecture_fingerprint(old) == old_fingerprint
    # Legacy nested class name is part of recorded architecture fingerprints.
    assert type(old).__qualname__ == "_build_factorized_band_operator.<locals>.FactorizedBandOperator"
    clone = build_model("global_local_channel_gate_v1", **args).eval()
    clone.load_state_dict(new.state_dict(), strict=True)
    torch.testing.assert_close(clone(x), new(x))


def test_formal_gl_parameter_delta():
    pytest.importorskip("torch")
    model = build_model("global_local_channel_gate_v1")
    assert sum(p.numel() for p in model.parameters()) == 2572746


@pytest.mark.parametrize("name", ["official_fno_band_v1", "official_tucker_tfno_band_v1"])
def test_official_adapter_coverage_roundtrip_and_gradient(name):
    torch = pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    args = dict(input_shape=(1, 8, 8), output_shape=(3, 2), width=8, depth=2, modes=(2, 2))
    model = build_model(name, **args).eval()
    conv = model.backbone.fno_blocks.convs[0]
    assert tuple(conv.n_modes) == (4, 2)
    assert tuple(conv.weight.shape) == (8, 8, 4, 2)
    x = torch.randn(2, 1, 8, 8)
    y = model(x)
    assert y.shape == (2, 3, 2) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    clone = build_model(name, **args).eval()
    clone.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(clone(x), y)
