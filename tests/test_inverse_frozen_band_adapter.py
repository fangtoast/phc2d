"""Test frozen checkpoint reconstruction and input gradients.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from phc2d.inverse.frozen_band_adapter import FrozenBandAdapter
from phc2d.models import build_model


def checkpoint(tmp_path):
    model = build_model("mlp", input_shape=(2,), output_shape=(2, 2))
    payload = dict(model_name="mlp", model_params={}, input_shape=(2,), output_shape=(2, 2),
                   state_dict=model.state_dict(), target_mean=torch.ones(2, 2),
                   target_std=torch.full((2, 2), 2.0))
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), model


def test_prediction_and_gradient(tmp_path):
    path, digest, model = checkpoint(tmp_path)
    model.eval()
    adapter = FrozenBandAdapter(path, sha256=digest)
    x = torch.tensor([[0.3, 0.4]], requires_grad=True)
    expected = model(x).reshape(1, 2, 2) * 2 + 1
    actual = adapter.predict_tensor(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(not p.requires_grad for p in adapter.model.parameters())


def test_rejects_wrong_hash_and_shape(tmp_path):
    path, digest, _ = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        FrozenBandAdapter(path, sha256="0" * 64)
    adapter = FrozenBandAdapter(path, sha256=digest)
    with pytest.raises(ValueError, match="input shape"):
        adapter.predict_tensor(torch.ones(1, 3))


def test_rejects_corrupt_k_identity(tmp_path):
    path, _, _ = checkpoint(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["k_points"] = torch.zeros(2, 2)
    payload["k_points_sha256"] = "0" * 64
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="k-point SHA-256"):
        FrozenBandAdapter(path, sha256=digest)
