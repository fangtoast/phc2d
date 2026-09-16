"""Tests for the project-native Full-Wh model upgrades.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import pytest

from phc2d.models import build_model, get_model_spec


UPGRADE_NAMES = (
    "mlp_mixer_band_v1",
    "res_mlp_gated_v2",
    "convnext_band_v1",
    "tfno2d_band_v3",
    "dual_lane_band_operator_v1",
    "global_local_band_operator_v1",
)


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    import torch

    return torch


def _small_case(name: str, *, head_kind: str = "dense") -> tuple[dict, tuple[int, ...]]:
    common = {"output_shape": (5, 3), "head_kind": head_kind}
    if name == "mlp_mixer_band_v1":
        return {
            **common,
            "input_shape": (32, 32),
            "token_grid": (8, 8),
            "embed_dim": 32,
            "depth": 2,
            "token_mlp_dim": 16,
            "channel_mlp_dim": 64,
        }, (2, 1, 32, 32)
    if name == "res_mlp_gated_v2":
        return {
            **common,
            "input_shape": (32 * 32,),
            "hidden_dim": 32,
            "depth": 2,
        }, (2, 32 * 32)
    if name == "convnext_band_v1":
        return {
            **common,
            "input_shape": (32, 32),
            "stage_dims": (8, 16, 32),
            "stage_depths": (1, 1, 1),
        }, (2, 1, 32, 32)
    return {
        **common,
        "input_shape": (32, 32),
        "width": 8,
        "depth": 2,
        "modes": (4, 4),
        "norm_groups": 2,
    }, (2, 1, 32, 32)


def test_upgrade_registry_contracts_are_explicit() -> None:
    specs = {name: get_model_spec(name) for name in UPGRADE_NAMES}
    assert specs["res_mlp_gated_v2"].input_layout == "flat"
    assert all(spec.accepts_k_points for spec in specs.values())
    assert specs["dual_lane_band_operator_v1"].accepts_lane_id is True
    assert all(
        not spec.accepts_lane_id
        for name, spec in specs.items()
        if name != "dual_lane_band_operator_v1"
    )


def test_global_local_bridge_has_fixed_spectral_and_periodic_large_kernel_branches() -> None:
    torch = _require_torch()
    model = build_model(
        "global_local_band_operator_v1",
        input_shape=(17, 19),
        output_shape=(5, 3),
        width=8,
        depth=2,
        modes=(4, 4),
        norm_groups=2,
        channel_expansion=2,
        local_kernel=7,
    )
    block = model.blocks[0]
    assert block.active_branches == ("spectral", "local")
    assert block.gates is None
    assert block.channel is None
    assert block.local[0].groups == 8
    assert block.local[0].padding_mode == "circular"
    assert block.local[0].kernel_size == (7, 7)

    model.eval()
    x = torch.randn(2, 17, 19)
    with torch.no_grad():
        output = model(x)
        shifted = model(torch.roll(x, shifts=(3, -4), dims=(-2, -1)))
    assert output.shape == (2, 5, 3)
    assert torch.allclose(output, shifted, atol=2.0e-5, rtol=2.0e-5)


def test_global_local_branch_interventions_preserve_raw_and_renorm_scales() -> None:
    torch = _require_torch()
    model = build_model(
        "global_local_band_operator_v1",
        input_shape=(8, 8),
        output_shape=(5, 3),
        width=4,
        depth=1,
        modes=(2, 2),
        norm_groups=1,
        channel_expansion=1,
        local_kernel=3,
        drop_path=0.0,
        dropout=0.0,
    ).eval()
    block = model.blocks[0]
    block.norm = torch.nn.Identity()
    block.spectral = torch.nn.Identity()
    block.local = torch.nn.Identity()
    block.drop_path = torch.nn.Identity()
    x = torch.randn(2, 4, 8, 8)

    expected_scales = {
        "full": 1.0,
        "g_off_raw": 0.5,
        "l_off_raw": 0.5,
        "g_off_renorm": 1.0,
        "l_off_renorm": 1.0,
    }
    for mode, scale in expected_scales.items():
        model.set_branch_intervention(mode)
        observed = block(x, None)
        expected = torch.nn.functional.gelu(x + scale * x)
        assert torch.equal(observed, expected)


def test_global_local_legacy_component_mask_maps_to_renormalized_branch() -> None:
    torch = _require_torch()
    model = build_model(
        "global_local_band_operator_v1",
        input_shape=(8, 8),
        output_shape=(5, 3),
        width=4,
        depth=1,
        modes=(2, 2),
        norm_groups=1,
        channel_expansion=1,
        local_kernel=3,
        drop_path=0.0,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 1, 8, 8)
    model.set_branch_intervention("g_off_renorm")
    with torch.no_grad():
        intervention = model(x)
    model.set_component_mask(("spectral",))
    with torch.no_grad():
        legacy = model(x)
    assert torch.equal(intervention, legacy)


def test_global_local_branch_intervention_does_not_change_state_dict_schema() -> None:
    _require_torch()
    model = build_model(
        "global_local_band_operator_v1",
        input_shape=(8, 8),
        output_shape=(5, 3),
        width=4,
        depth=1,
        modes=(2, 2),
        norm_groups=1,
    )
    before = tuple(model.state_dict())
    model.set_branch_intervention("l_off_raw")
    assert tuple(model.state_dict()) == before


@pytest.mark.parametrize("name", UPGRADE_NAMES)
@pytest.mark.parametrize("head_kind", ("dense", "k_conditioned"))
def test_upgrade_forward_backward_and_dynamic_k_shape(name: str, head_kind: str) -> None:
    torch = _require_torch()
    kwargs, x_shape = _small_case(name, head_kind=head_kind)
    default_k = torch.rand(5, 2)
    kwargs["k_points"] = default_k
    model = build_model(name, **kwargs)
    x = torch.randn(*x_shape, requires_grad=True)
    call_kwargs = {}
    if head_kind == "k_conditioned":
        call_kwargs["k_points"] = torch.rand(7, 2)
    if name == "dual_lane_band_operator_v1":
        call_kwargs["lane_id"] = torch.tensor([0, 1])
    output = model(x, **call_kwargs)
    expected_k = 7 if head_kind == "k_conditioned" else 5
    assert tuple(output.shape) == (2, expected_k, 3)
    output.square().mean().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())
    trainable_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_gradients
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in trainable_gradients
    )


def test_k_conditioned_head_follows_k_point_permutation() -> None:
    torch = _require_torch()
    kwargs, x_shape = _small_case("res_mlp_gated_v2", head_kind="k_conditioned")
    points = torch.rand(5, 2)
    model = build_model("res_mlp_gated_v2", **kwargs, k_points=points).eval()
    x = torch.randn(*x_shape)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        original = model(x, k_points=points)
        permuted = model(x, k_points=points[permutation])
    assert torch.allclose(permuted, original[:, permutation], atol=1.0e-6, rtol=1.0e-5)


def test_dual_lane_conditioning_is_required_and_changes_output() -> None:
    torch = _require_torch()
    kwargs, x_shape = _small_case("dual_lane_band_operator_v1")
    model = build_model("dual_lane_band_operator_v1", **kwargs).eval()
    x = torch.randn(*x_shape)
    with pytest.raises(ValueError, match="requires lane_id"):
        model(x)
    with pytest.raises(ValueError, match="values must be"):
        model(x, lane_id=torch.tensor([0, 2]))
    with torch.no_grad():
        binary = model(x, lane_id=torch.tensor([0, 0]))
        graded = model(x, lane_id=torch.tensor([1, 1]))
    assert not torch.allclose(binary, graded)


@pytest.mark.parametrize("resolution", (32, 64, 128))
@pytest.mark.parametrize("name", ("mlp_mixer_band_v1", "convnext_band_v1", "tfno2d_band_v3"))
def test_variable_grid_models_preserve_output_contract(name: str, resolution: int) -> None:
    torch = _require_torch()
    kwargs, _ = _small_case(name)
    kwargs["input_shape"] = (resolution, resolution)
    model = build_model(name, **kwargs).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, resolution, resolution))
    assert tuple(output.shape) == (1, 5, 3)


def test_tfno_has_independent_positive_and_negative_spectral_gradients() -> None:
    torch = _require_torch()
    kwargs, x_shape = _small_case("tfno2d_band_v3")
    model = build_model("tfno2d_band_v3", **kwargs)
    model(torch.randn(*x_shape)).square().mean().backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if ".spectral.positive.core" in name or ".spectral.negative.core" in name
    }
    assert gradients
    assert any("positive" in name for name in gradients)
    assert any("negative" in name for name in gradients)
    assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients.values())


def test_tfno_default_width_reconstructs_modes_without_large_intermediate() -> None:
    torch = _require_torch()
    model = build_model(
        "tfno2d_band_v3",
        input_shape=(16, 16),
        output_shape=(5, 3),
        width=64,
        depth=1,
        modes=(12, 12),
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 1, 16, 16))
    assert tuple(output.shape) == (1, 5, 3)
    assert bool(torch.isfinite(output).all())


@pytest.mark.parametrize("name", UPGRADE_NAMES)
@pytest.mark.parametrize("head_kind", ("dense", "k_conditioned"))
def test_default_parameter_budget_is_two_to_eight_million(name: str, head_kind: str) -> None:
    torch = _require_torch()
    model = build_model(
        name,
        head_kind=head_kind,
        k_points=torch.rand(81, 2) if head_kind == "k_conditioned" else None,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 2_000_000 <= parameter_count <= 8_000_000
    del model
    gc.collect()


def test_upgrade_checkpoint_restores_exactly(tmp_path: Path) -> None:
    torch = _require_torch()
    kwargs, x_shape = _small_case("convnext_band_v1", head_kind="k_conditioned")
    points = torch.rand(5, 2)
    source = build_model("convnext_band_v1", **kwargs, k_points=points).eval()
    checkpoint = tmp_path / "convnext_band_v1.pt"
    torch.save(source.state_dict(), checkpoint)
    restored = build_model("convnext_band_v1", **kwargs, k_points=points).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    x = torch.randn(*x_shape)
    with torch.no_grad():
        assert torch.equal(source(x), restored(x))
