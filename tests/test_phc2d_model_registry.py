"""Tests for Gate 2 model registry specs and CPU shape smoke.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phc2d.models import (
    compare_model_identities,
    describe_model_identity,
    build_model,
    get_canonical_model_name,
    get_model_factory,
    get_model_factory_identity,
    get_model_spec,
    list_model_specs,
    list_models,
    register_model,
    ModelSpec,
)


def test_phc2d_models_import_does_not_import_torch() -> None:
    code = (
        "import sys\n"
        "import phc2d\n"
        "import phc2d.models\n"
        "print('torch' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_builtin_model_specs_are_registered() -> None:
    names = set(list_models())
    assert {"mlp", "res_mlp", "pod_res_mlp", "lifted_cnn", "fno_lite", "fno2d_band"}.issubset(names)
    assert {
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
        "dual_lane_band_operator_v1",
    }.issubset(names)

    specs = {spec.name: spec for spec in list_model_specs()}
    assert specs["mlp"].alias_of is None
    assert specs["mlp"].canonical_name == "mlp"
    assert specs["mlp"].input_repr == "rho_wedge"
    assert specs["mlp"].output_target == "Wh"
    assert specs["res_mlp"].family == "mlp"
    assert specs["res_mlp"].alias_of is None
    assert specs["res_mlp"].input_repr == "rho_wedge"
    assert specs["res_mlp"].output_target == "Wh"
    assert specs["lifted_cnn"].supports_variable_grid is True
    assert specs["fno_lite"].family == "neural_operator"
    assert specs["fno2d_band"].input_repr == "epsilon_grid"
    assert specs["fno2d_band"].output_target == "Wh"
    assert "torch" in specs["fno_lite"].requires
    assert specs["res_mlp_gated_v2"].input_layout == "flat"
    assert specs["tfno2d_band_v3"].accepts_k_points is True
    assert specs["dual_lane_band_operator_v1"].accepts_lane_id is True

    placeholder = get_model_spec("pod_res_mlp")
    assert "pod_basis" in placeholder.requires


def test_alias_model_spec_must_declare_alias_of() -> None:
    spec = get_model_spec("res_mlp")

    def _alias_factory(**kwargs):
        return build_model("res_mlp", **kwargs)

    register_model(
        "unit_res_mlp_alias",
        _alias_factory,
        replace=True,
        spec=ModelSpec(
            name="unit_res_mlp_alias",
            family=spec.family,
            input_repr=spec.input_repr,
            output_target=spec.output_target,
            canonical_name="res_mlp",
            alias_of="res_mlp",
            requires=spec.requires,
            defaults=spec.defaults,
            source="unit test alias",
        ),
    )
    alias_spec = get_model_spec("unit_res_mlp_alias")
    assert alias_spec.alias_of == "res_mlp"
    assert get_canonical_model_name("unit_res_mlp_alias") == "res_mlp"


def test_build_model_merges_spec_defaults_and_allows_overrides() -> None:
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(ImportError, match="torch"):
            build_model("res_mlp")
        return

    model = build_model("res_mlp", input_shape=(7,), output_shape=(4, 2), hidden_dim=11)
    assert model.input_shape == (7,)
    assert model.output_shape == (4, 2)


def test_plain_mlp_is_parameter_matched_but_not_a_res_mlp_alias() -> None:
    if importlib.util.find_spec("torch") is None:
        return

    import torch

    torch.manual_seed(17)
    plain = build_model("mlp")
    residual = build_model("res_mlp_wide")
    assert sum(param.numel() for param in plain.parameters()) == 878_646
    assert sum(param.numel() for param in plain.parameters()) == sum(
        param.numel() for param in residual.parameters()
    )
    assert get_model_factory_identity("mlp") != get_model_factory_identity("res_mlp_wide")
    assert get_model_spec("mlp").alias_of is None
    assert get_canonical_model_name("mlp") == "mlp"

    residual.load_state_dict(plain.state_dict())
    plain.eval()
    residual.eval()
    x = torch.randn(4, 36)
    with torch.no_grad():
        plain_output = plain(x)
        residual_output = residual(x)
    assert tuple(plain_output.shape) == (4, 31, 10)
    assert not torch.allclose(plain_output, residual_output)

    plain_identity = describe_model_identity(plain, "mlp")
    residual_identity = describe_model_identity(residual, "res_mlp_wide")
    comparison = compare_model_identities(plain_identity, residual_identity)
    assert comparison["same_architecture_fingerprint"] is False
    assert comparison["same_factory_function_identity"] is False


def test_plain_mlp_checkpoint_state_restores_exactly(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        return

    import torch

    torch.manual_seed(23)
    model_kwargs = {
        "input_shape": (7,),
        "output_shape": (4, 2),
        "hidden_dim": 11,
        "depth": 2,
        "dropout": 0.05,
    }
    source = build_model("mlp", **model_kwargs)
    checkpoint = tmp_path / "mlp.pt"
    torch.save(source.state_dict(), checkpoint)
    restored = build_model("mlp", **model_kwargs)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    source.eval()
    restored.eval()
    x = torch.randn(3, 7)
    with torch.no_grad():
        assert torch.equal(source(x), restored(x))


def test_pod_res_mlp_placeholder_is_explicit() -> None:
    with pytest.raises(NotImplementedError, match="POD basis"):
        build_model("pod_res_mlp")


@pytest.mark.parametrize(
    ("name", "input_shape", "x_shape", "output_shape", "overrides"),
    [
        ("mlp", (7,), (3, 7), (4, 2), {"hidden_dim": 13, "depth": 2}),
        ("res_mlp", (7,), (3, 7), (4, 2), {"hidden_dim": 13, "depth": 2}),
        ("lifted_cnn", (1, 11, 13), (3, 1, 11, 13), (4, 2), {"channels": 9}),
        (
            "fno_lite",
            (1, 11, 13),
            (3, 1, 11, 13),
            (4, 2),
            {"width": 9, "depth": 2, "modes": (3, 4)},
        ),
        (
            "fno2d_band",
            (1, 11, 13),
            (3, 1, 11, 13),
            (4, 2),
            {"width": 9, "depth": 2, "modes": (3, 4)},
        ),
    ],
)
def test_torch_baseline_cpu_forward_shapes(
    name: str,
    input_shape: tuple[int, ...],
    x_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    overrides: dict[str, int | tuple[int, int]],
) -> None:
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(ImportError, match="torch"):
            build_model(name, input_shape=input_shape, output_shape=output_shape, **overrides)
        return

    import torch

    model = build_model(name, input_shape=input_shape, output_shape=output_shape, **overrides)
    model.eval()
    with torch.no_grad():
        y = model(torch.zeros(x_shape, dtype=torch.float32))
    assert tuple(y.shape) == (x_shape[0], *output_shape)


@pytest.mark.parametrize(
    ("name", "input_shape", "x_shape", "output_shape", "overrides"),
    [
        ("mlp", (36,), (4, 36), (31, 10), {"hidden_dim": 16, "depth": 2}),
        ("res_mlp", (36,), (4, 36), (31, 10), {"hidden_dim": 16, "depth": 2}),
        ("lifted_cnn", (16, 16), (4, 16, 16), (31, 10), {"channels": 8}),
        ("lifted_cnn", (1, 16, 16), (4, 1, 16, 16), (31, 10), {"channels": 8}),
        (
            "fno_lite",
            (1, 16, 16),
            (4, 1, 16, 16),
            (31, 10),
            {"width": 8, "depth": 2, "modes": (6, 6)},
        ),
        (
            "fno2d_band",
            (1, 16, 16),
            (4, 1, 16, 16),
            (31, 10),
            {"width": 8, "depth": 2, "modes": (6, 6)},
        ),
    ],
)
def test_layer2_paper_like_schema_shapes_are_not_hardcoded(
    name: str,
    input_shape: tuple[int, ...],
    x_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    overrides: dict[str, int | tuple[int, int]],
) -> None:
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(ImportError, match="torch"):
            build_model(name, input_shape=input_shape, output_shape=output_shape, **overrides)
        return

    import torch

    model = build_model(name, input_shape=input_shape, output_shape=output_shape, **overrides)
    model.eval()
    with torch.no_grad():
        y = model(torch.zeros(x_shape, dtype=torch.float32))
    assert tuple(y.shape) == (x_shape[0], *output_shape)


def test_model_identity_fingerprints_disambiguate_fno_variants() -> None:
    if importlib.util.find_spec("torch") is None:
        return

    import torch

    shared_kwargs = {
        "input_shape": (1, 11, 13),
        "output_shape": (4, 2),
        "width": 8,
        "depth": 2,
        "modes": (3, 4),
    }
    fno2d_kwargs = {**shared_kwargs, "band_modes": (3, 4), "fourier_rank": 4}
    fno_lite_kwargs = dict(shared_kwargs)

    torch.manual_seed(0)
    model_lite = build_model("fno_lite", **fno_lite_kwargs)
    torch.manual_seed(0)
    model_band = build_model("fno2d_band", **fno2d_kwargs)
    torch.manual_seed(0)
    model_band_same = build_model("fno2d_band", **fno2d_kwargs)

    lite_module = type(model_lite)
    band_module = type(model_band)
    assert lite_module is not band_module
    assert get_model_factory_identity("fno_lite") != get_model_factory_identity("fno2d_band")
    assert get_canonical_model_name("fno_lite") == "fno_lite"
    assert get_canonical_model_name("fno2d_band") == "fno2d_band"
    identity_lite = describe_model_identity(model_lite, "fno_lite", model_kwargs=fno_lite_kwargs)
    identity_band = describe_model_identity(model_band, "fno2d_band", model_kwargs=fno2d_kwargs)
    identity_band_same = describe_model_identity(model_band_same, "fno2d_band", model_kwargs=fno2d_kwargs)
    comparison = compare_model_identities(identity_lite, identity_band)
    assert comparison["all_identity_fields_match"] is False
    assert comparison["same_architecture_fingerprint"] is False
    assert comparison["same_state_dict_schema_hash"] is False
    assert identity_lite["state_dict_schema_hash"] != identity_band["state_dict_schema_hash"]
    assert sum(p.numel() for p in model_lite.parameters()) != sum(p.numel() for p in model_band.parameters())
    assert identity_band["state_dict_schema_hash"] == identity_band_same["state_dict_schema_hash"]

    x = torch.randn(2, 1, 11, 13)
    model_lite.eval()
    model_band.eval()
    with torch.no_grad():
        y_lite = model_lite(x)
        y_band = model_band(x)
    assert not torch.equal(y_lite, y_band)


def test_model_identity_payload_fields_for_forward_baselines() -> None:
    if importlib.util.find_spec("torch") is None:
        return

    names = ["res_mlp", "lifted_cnn", "fno_lite", "fno2d_band"]
    for name in names:
        kwargs = {
            "input_shape": (1, 11, 13) if name in {"lifted_cnn", "fno_lite", "fno2d_band"} else (7,),
            "output_shape": (4, 2) if name in {"lifted_cnn", "fno_lite", "fno2d_band"} else (4, 2),
            "hidden_dim": 11 if name == "res_mlp" else 8,
            "depth": 2,
            "channels": 6 if name == "lifted_cnn" else None,
            "width": 8 if name in {"fno_lite", "fno2d_band"} else None,
            "modes": (3, 4) if name in {"fno_lite", "fno2d_band"} else None,
            "band_modes": (3, 4) if name == "fno2d_band" else None,
            "fourier_rank": 4 if name == "fno2d_band" else None,
        }
        filtered = {k: v for k, v in kwargs.items() if v is not None}

        model = build_model(name, **filtered)
        factory = get_model_factory(name)

        assert get_model_spec(name).canonical_name == get_canonical_model_name(name)
        assert model.__class__.__module__ == "phc2d.models.baselines"
        assert model.__class__.__name__ != ""
        assert callable(factory)
        assert get_model_factory_identity(name)

        identity = describe_model_identity(model, name, model_kwargs=filtered)
        assert identity["requested_model_name"] == name
        assert identity["canonical_model_name"] == name
        assert identity["model_spec"]["name"] == name
        assert identity["state_dict_key_names"]
        assert isinstance(identity["state_dict_schema_hash"], str) and len(identity["state_dict_schema_hash"]) == 64
        assert isinstance(identity["architecture_fingerprint"], str) and len(identity["architecture_fingerprint"]) == 64
        assert identity["parameter_count"] == identity["trainable_parameter_count"]
