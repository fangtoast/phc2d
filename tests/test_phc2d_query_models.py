"""Tests for Full-2A query model registry entries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util

import pytest

from phc2d.models import build_model, describe_model_identity, get_model_spec, list_models
from phc2d.query.runner import _is_verified_k_path_status, _learning_rate_label, _learning_rates


def test_query_model_specs_are_registered():
    names = set(list_models())
    assert {"mlp_query", "deep_onet_query", "fno2d_query"}.issubset(names)
    for name in ("mlp_query", "deep_onet_query", "fno2d_query"):
        spec = get_model_spec(name)
        assert spec.supports_query is True
        assert spec.output_target == "band_query"


def test_formal_v2_model_specs_are_registered():
    names = set(list_models())
    expected = {"res_mlp_wide", "fno2d_band_v2", "deep_onet_query_v2", "fno2d_query_v2"}
    assert expected.issubset(names)
    assert get_model_spec("res_mlp_wide").output_target == "Wh"
    assert get_model_spec("fno2d_band_v2").input_repr == "epsilon_grid"
    assert get_model_spec("deep_onet_query_v2").supports_query is True
    assert get_model_spec("fno2d_query_v2").supports_query is True


def test_query_runner_accepts_documented_numerical_tolerance_k_path_status():
    assert _is_verified_k_path_status("physically_verified") is True
    assert _is_verified_k_path_status("physically_verified_with_documented_numerical_tolerance") is True
    assert _is_verified_k_path_status("pilot_generated") is False


def test_query_runner_parses_learning_rate_grid():
    raw = {"training": {"learning_rates": [0.001, 0.0003, 0.0001]}}

    assert _learning_rates(raw) == [0.001, 0.0003, 0.0001]
    assert _learning_rate_label(0.001) == "lr1em03"


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("mlp_query", {"hidden_dim": 16, "depth": 2}),
        ("deep_onet_query", {"branch_width": 16, "trunk_width": 16, "latent_dim": 8, "depth": 2}),
        ("fno2d_query", {"width": 8, "depth": 2, "modes": (3, 4), "query_hidden_dim": 8}),
        ("deep_onet_query_v2", {"branch_width": 16, "trunk_width": 16, "latent_dim": 8, "depth": 2}),
        ("fno2d_query_v2", {"width": 8, "depth": 2, "modes": (3, 4), "query_hidden_dim": 8}),
    ],
)
def test_query_model_forward_shapes(name, kwargs):
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(ImportError, match="torch"):
            build_model(name, input_shape=(1, 6, 6), output_shape=(3,), k_dim=1, **kwargs)
        return

    import torch

    model = build_model(name, input_shape=(1, 6, 6), output_shape=(3,), k_dim=1, **kwargs)
    y = model(torch.zeros((4, 6, 6), dtype=torch.float32), torch.zeros((4, 1), dtype=torch.float32))
    assert tuple(y.shape) == (4, 3)


def test_query_model_identity_fingerprints_are_independent():
    if importlib.util.find_spec("torch") is None:
        return

    kwargs = {"input_shape": (1, 6, 6), "output_shape": (3,), "k_dim": 1}
    models = {
        "mlp_query": build_model("mlp_query", **kwargs, hidden_dim=16, depth=2),
        "deep_onet_query": build_model(
            "deep_onet_query",
            **kwargs,
            branch_width=16,
            trunk_width=16,
            latent_dim=8,
            depth=2,
        ),
        "fno2d_query": build_model(
            "fno2d_query",
            **kwargs,
            width=8,
            depth=2,
            modes=(3, 4),
            query_hidden_dim=8,
        ),
    }
    fingerprints = {
        name: describe_model_identity(model, name, model_kwargs=kwargs)["architecture_fingerprint"]
        for name, model in models.items()
    }
    assert len(set(fingerprints.values())) == 3
