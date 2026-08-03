"""Tests for B3 active-component field data and losses.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from phc2d.forward.field_eigen_v2 import _data_loss, _epsilon_on_field_grid_numpy, _history_best_and_stale, _lambda_at_epoch, _latest_epoch_checkpoint, _load_dense_frequency_supervision, _periodic_grid_torch, _phase_align_numpy, _rayleigh_consistency_loss, _read_history, _relative_l2_numpy, _validate_p6_protocol, active_channels_to_complex, complex_to_active_channels, load_field_eigen_v2, phase_invariant_field_loss, physics_frequency_for_residual, physics_frequency_from_prediction, reconstruct_full_te_fields


def test_active_te_codec_roundtrip_and_zero_reconstruction() -> None:
    rng = np.random.default_rng(4)
    active = rng.standard_normal((3, 7, 5, 3)) + 1j * rng.standard_normal((3, 7, 5, 3))
    channels = complex_to_active_channels(active)
    restored = active_channels_to_complex(channels)
    efield, hfield = reconstruct_full_te_fields(restored)
    assert np.allclose(restored, active)
    assert np.all(efield[..., 0, 2] == 0)
    assert np.all(hfield[..., 0, :2] == 0)
    assert np.allclose(efield[..., 0, :2], active[..., :2])
    assert np.allclose(hfield[..., 0, 2], active[..., 2])


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_phase_invariant_loss_uses_one_global_phase() -> None:
    import torch

    target_complex = np.ones((2, 5, 7, 3), dtype=np.complex64) * (1.0 + 2.0j)
    target = torch.tensor(complex_to_active_channels(target_complex))
    phase = np.exp(1j * 0.73)
    prediction = torch.tensor(complex_to_active_channels(target_complex * phase))
    total, e_loss, h_loss, _ = phase_invariant_field_loss(
        torch, prediction, target, epsilon=torch.ones((2, 5, 7))
    )
    assert float(total) == pytest.approx(0.0, abs=1.0e-10)
    assert float(e_loss) == pytest.approx(0.0, abs=1.0e-10)
    assert float(h_loss) == pytest.approx(0.0, abs=1.0e-10)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_phase_alignment_uses_epsilon_as_the_electric_metric() -> None:
    import torch

    target_complex = np.zeros((1, 1, 2, 3), dtype=np.complex64)
    target_complex[..., 0] = 1.0
    target_complex[..., 2] = 1.0
    prediction_complex = target_complex.copy()
    prediction_complex[0, 0, 0, 0] = 1.0j
    prediction_complex[0, 0, 1, 0] = -1.0j
    epsilon = np.asarray([[[9.0, 1.0]]], dtype=np.float32)

    _, e_loss, _, _ = phase_invariant_field_loss(
        torch,
        torch.tensor(complex_to_active_channels(prediction_complex)),
        torch.tensor(complex_to_active_channels(target_complex)),
        epsilon=torch.tensor(epsilon),
        e_weight=1.0,
        h_weight=0.0,
    )
    assert float(e_loss) == pytest.approx(0.4)

    aligned = _phase_align_numpy(prediction_complex, target_complex, epsilon)
    metric = _epsilon_on_field_grid_numpy(epsilon, (1, 2))
    error = _relative_l2_numpy(target_complex[..., :2], aligned[..., :2], metric)
    assert float(error[0]) == pytest.approx(np.sqrt(0.4))


@pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py is optional")
def test_loader_preserves_source_split_and_excludes_frequency_from_inputs(tmp_path) -> None:
    import h5py

    path = tmp_path / "field.h5"
    count, nx, ny = 6, 8, 12
    efield = np.zeros((count, nx, ny, 1, 3), dtype=np.complex64); efield[..., 0, 0] = 1.0
    hfield = np.zeros_like(efield); hfield[..., 0, 2] = 2.0j
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("efield_complex", data=efield); handle.create_dataset("hfield_complex", data=hfield)
        handle.create_dataset("epsilon", data=np.ones((count, 2, 3), dtype=np.float32))
        handle.create_dataset("k_points", data=np.zeros((count, 2), dtype=np.float32)); handle.create_dataset("band_index", data=np.arange(2, 8)); handle.create_dataset("eigenfrequency", data=np.linspace(0.2, 0.8, count))
        handle.create_dataset("dx", data=np.full(count, 1 / nx)); handle.create_dataset("dy", data=np.full(count, 1 / ny))
        handle.create_dataset("source_split", data=np.asarray(["train"] * 2 + ["val"] * 2 + ["test"] * 2, dtype=object), dtype=string)
        handle.create_dataset("record_id", data=np.asarray([f"r{i}" for i in range(count)], dtype=object), dtype=string)
        handle.create_dataset("source_sample_id", data=np.asarray([f"s{i}" for i in range(count)], dtype=object), dtype=string)
        handle.create_dataset("polarization", data=np.asarray(["TE"] * count, dtype=object), dtype=string)
    data = load_field_eigen_v2(path, {"representation": {"periodic_coordinates": True}, "normalization": {"epsilon_range": [1.0, 8.9], "k_scale": 0.5, "band_range_one_based": [2, 7]}, "dataset": {"expected_split_records": {"train": 2, "val": 2, "test": 2}}})
    assert data.grid.shape == (count, 5, nx, ny)
    assert data.context.shape == (count, 3)
    assert data.target.shape == (count, 6, nx, ny)
    assert {name: len(index) for name, index in data.indices.items()} == {"train": 2, "val": 2, "test": 2}
    assert data.grid.shape[1] == 5 and data.context.shape[1] == 3


@pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py is optional")
def test_dense_frequency_supervision_uses_only_b3_train_sources(tmp_path: Path, monkeypatch) -> None:
    import h5py
    import phc2d.forward.field_eigen_v2 as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    path = tmp_path / "field.h5"; count, nx, ny = 6, 8, 8
    efield = np.zeros((count, nx, ny, 1, 3), dtype=np.complex64); efield[..., 0, 0] = 1.0
    hfield = np.zeros_like(efield); hfield[..., 0, 2] = 1.0
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("efield_complex", data=efield); handle.create_dataset("hfield_complex", data=hfield)
        handle.create_dataset("epsilon", data=np.ones((count, 2, 2), dtype=np.float32)); handle.create_dataset("k_points", data=np.zeros((count, 2), dtype=np.float32)); handle.create_dataset("band_index", data=np.full(count, 2)); handle.create_dataset("eigenfrequency", data=np.full(count, 0.3)); handle.create_dataset("dx", data=np.full(count, 1 / nx)); handle.create_dataset("dy", data=np.full(count, 1 / ny)); handle.create_dataset("source_split", data=np.asarray(["train", "train", "val", "val", "test", "test"], dtype=object), dtype=string); handle.create_dataset("record_id", data=np.asarray([f"r{i}" for i in range(count)], dtype=object), dtype=string); handle.create_dataset("source_sample_id", data=np.asarray([f"dataset_{i}" for i in range(count)], dtype=object), dtype=string); handle.create_dataset("polarization", data=np.asarray(["TE"] * count, dtype=object), dtype=string)
    v2_path = tmp_path / "v2.npz"; np.savez(v2_path, Wh=np.full((count, 3, 4), 0.3, dtype=np.float32), idx_train=np.asarray([0, 1], dtype=np.int64))
    sidecar_path = tmp_path / "sidecar.json"; sidecar_path.write_text(json.dumps({"Nk": 3, "k_points": [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]]}), encoding="utf-8")
    cfg = {"representation": {"periodic_coordinates": True}, "normalization": {"epsilon_range": [1.0, 8.9], "k_scale": 0.5, "band_range_one_based": [2, 4]}, "dataset": {"expected_split_records": {"train": 2, "val": 2, "test": 2}}, "training": {"dense_frequency_supervision": {"enabled": True, "v2_frequency_dataset_path": "v2.npz", "v2_frequency_dataset_sha256": module._hash(v2_path), "k_path_sidecar_path": "sidecar.json", "k_path_sidecar_sha256": module._hash(sidecar_path)}}}
    dense = _load_dense_frequency_supervision(load_field_eigen_v2(path, cfg), cfg)
    assert dense is not None and dense.source_ids.tolist() == ["dataset_0", "dataset_1"]
    assert dense.context.shape == (2 * 3 * 3, 3) and dense.frequency.shape == (18,)
    cfg["training"]["dense_frequency_supervision"]["v2_frequency_dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        _load_dense_frequency_supervision(load_field_eigen_v2(path, cfg), cfg)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_multiresolution_grid_and_rayleigh_consistency_are_differentiable() -> None:
    import torch
    from phc2d.physics import evaluate_te_bloch_eigen_residual_torch

    epsilon = torch.ones((2, 2, 2), dtype=torch.float32)
    grid = _periodic_grid_torch(torch, epsilon, resolution=8, epsilon_range=(1.0, 8.9))
    assert grid.shape == (2, 5, 8, 8)
    field = torch.complex(torch.randn((2, 8, 8, 2), requires_grad=True), torch.randn((2, 8, 8, 2), requires_grad=True))
    frequency = torch.tensor([0.3, 0.35], requires_grad=True)
    residual = evaluate_te_bloch_eigen_residual_torch(field, epsilon=epsilon, eigenfrequency=frequency, dx=torch.full((2,), 1 / 8), dy=torch.full((2,), 1 / 8), bloch_k_point=torch.zeros((2, 2)))
    loss = _rayleigh_consistency_loss(torch, physical_frequency=frequency, residual=residual)
    loss.backward()
    assert torch.isfinite(loss) and frequency.grad is not None and torch.isfinite(frequency.grad).all()


def test_pino_sweep_uses_frozen_data_only_checkpoint() -> None:
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "research" / "field_eigen_te_pino_lambda_sweep_9216.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reference = raw["benchmark"]["cases"][0]
    assert reference["name"] == "fno_data_only_reference"
    assert reference["train"] is False
    assert reference["checkpoint_path"].endswith("/cases/fno_field_v2_seed0/model.pt")
    assert reference["training"] == {"regime": "data_only", "lambda_pde": 0.0}


def test_pino_lambda_schedule_respects_warmup_and_linear_ramp() -> None:
    target = 1.0e-2
    assert _lambda_at_epoch(1, target, 20, 20) == 0.0
    assert _lambda_at_epoch(20, target, 20, 20) == 0.0
    assert _lambda_at_epoch(21, target, 20, 20) == pytest.approx(0.05 * target)
    assert _lambda_at_epoch(30, target, 20, 20) == pytest.approx(0.5 * target)
    assert _lambda_at_epoch(40, target, 20, 20) == pytest.approx(target)
    assert _lambda_at_epoch(200, target, 20, 20) == pytest.approx(target)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_frequency_guard_detach_preserves_residual_value_and_blocks_frequency_gradient() -> None:
    import torch

    frequency = torch.tensor([0.3], requires_grad=True)
    field = torch.tensor([2.0], requires_grad=True)
    full = (field * physics_frequency_from_prediction(frequency, mode="full")).square().mean()
    detached = (field * physics_frequency_from_prediction(frequency, mode="detach")).square().mean()
    assert float(full.detach()) == pytest.approx(float(detached.detach()))
    full_frequency_gradient, full_field_gradient = torch.autograd.grad(full, (frequency, field), retain_graph=True)
    detached_frequency_gradient, detached_field_gradient = torch.autograd.grad(detached, (frequency, field), allow_unused=True)
    assert float(full_frequency_gradient.abs()) > 0.0 and float(full_field_gradient.abs()) > 0.0
    assert detached_frequency_gradient is None
    assert detached_field_gradient is not None and float(detached_field_gradient.abs()) > 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_data_loss_reads_frequency_and_equal_eh_weights() -> None:
    import torch

    target = torch.zeros((1, 6, 4, 4))
    prediction = torch.ones_like(target)
    freq = torch.tensor([1.0])
    total, e_loss, h_loss, frequency_loss = _data_loss(
        torch,
        prediction,
        freq,
        target,
        torch.zeros_like(freq),
        epsilon=torch.ones((1, 4, 4)),
        loss_weights={"field": 1.0, "frequency": 0.5, "e": 0.5, "h": 0.5},
    )
    assert float(e_loss) == pytest.approx(float(h_loss))
    assert float(total) == pytest.approx(float(e_loss + 0.5 * frequency_loss))


def test_formal_pino_config_freezes_selection_and_reuses_only_seed_zero() -> None:
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "research" / "field_eigen_te_pino_9216.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["training"]["regime"] == "pino_adapted"
    assert raw["training"]["lambda_pde"] == 0.01
    assert raw["pino_selection"]["selection_split"] == "val"
    assert len(raw["pino_selection"]["lambda_selection_sha256"]) == 64
    cases = raw["benchmark"]["cases"]
    assert cases[0]["name"] == "pino_adapted_seed0" and cases[0]["train"] is False
    assert cases[0]["checkpoint_path"].endswith("/pino_lambda_1e-2/model.pt")
    assert all("train" not in case for case in cases[1:])


def test_b4_frequency_guard_config_has_fixed_epochs_and_audited_cases() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "research" / "field_eigen_te_pino_frequency_guard_9216.yaml").read_text(encoding="utf-8"))
    training = raw["training"]
    assert training["epochs"] == 200 and training["early_stopping_patience"] == 0
    assert training["checkpoint_epochs"] == [40, 80, 120, 160, 200]
    assert training["gradient_audit_epochs"] == [40, 80, 120, 160, 200]
    assert raw["benchmark"]["selection_rule"] == "pino_frequency_guard_validation"
    assert raw["benchmark"]["cases"][1]["b4_audited"] is False


def test_b4_repair_config_keeps_test_out_and_freezes_the_two_candidates() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "research" / "field_eigen_te_pino_b4_repair_validation_9216.yaml").read_text(encoding="utf-8"))
    training = raw["training"]
    assert raw["evaluation"]["split"] == "val"
    assert training["physics_frequency_gradient"] == "detach" and training["physics_resolutions"] == [64, 96, 128]
    assert training["dense_frequency_supervision"]["enabled"] is True
    cases = raw["benchmark"]["cases"]
    assert {case["training"]["rayleigh_consistency_weight"] for case in cases} == {0.0, 0.05}
    assert {case["training"]["seed"] for case in cases} == {0, 1, 2}


def test_physics_frequency_source_uses_detached_target_without_changing_predicted_modes() -> None:
    torch = pytest.importorskip("torch")
    predicted = torch.tensor([1.0], requires_grad=True)
    target = torch.tensor([2.0], requires_grad=True)
    assert physics_frequency_for_residual(
        predicted,
        target,
        source="predicted",
        gradient_mode="full",
    ) is predicted
    detached_prediction = physics_frequency_for_residual(
        predicted,
        target,
        source="predicted",
        gradient_mode="detach",
    )
    assert detached_prediction.item() == pytest.approx(1.0)
    assert detached_prediction.requires_grad is False
    target_value = physics_frequency_for_residual(
        predicted,
        target,
        source="target",
        gradient_mode="full",
    )
    assert target_value.item() == pytest.approx(2.0)
    assert target_value.requires_grad is False
    with pytest.raises(ValueError, match="source"):
        physics_frequency_for_residual(
            predicted,
            target,
            source="invalid",
            gradient_mode="detach",
        )


def test_p6_config_is_a_strict_target_frequency_ablation() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    p4_p5 = yaml.safe_load(
        (root / "configs" / "research" / "field_eigen_te_pino_b4_repair_validation_9216.yaml").read_text(encoding="utf-8")
    )
    p6 = yaml.safe_load(
        (root / "configs" / "research" / "field_eigen_te_pino_p6_target_frequency_9216.yaml").read_text(encoding="utf-8")
    )
    assert p6["evaluation"]["split"] == "val"
    assert p6["training"]["physics_frequency_source"] == "target"
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "minimum_learning_rate",
        "weight_decay",
        "lr_warmup_epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
        "physics_warmup_epochs",
        "physics_ramp_epochs",
        "lambda_pde",
        "loss_weights",
        "physics_frequency_gradient",
        "physics_resolutions",
        "physics_solver_floor_factors",
        "dense_frequency_supervision",
        "checkpoint_epochs",
        "gradient_audit_epochs",
        "select_model",
    ):
        assert p6["training"][key] == p4_p5["training"][key]
    assert p6["model"] == p4_p5["model"]
    assert p6["dataset"] == p4_p5["dataset"]
    assert p6["normalization"] == p4_p5["normalization"]
    assert p6["representation"] == p4_p5["representation"]
    cases = p6["benchmark"]["cases"]
    assert [case["name"] for case in cases[:3]] == [
        "P6A_target_pde_seed0",
        "P6A_target_pde_seed1",
        "P6A_target_pde_seed2",
    ]
    assert [case["training"]["rayleigh_consistency_weight"] for case in cases] == [0.0, 0.0, 0.0, 0.05, 0.05, 0.05]
    assert [case["training"]["seed"] for case in cases] == [0, 1, 2, 0, 1, 2]
    lineage = _validate_p6_protocol(p6)
    assert lineage["only_named_change"] == "residual_frequency_source_predicted_to_target"
    assert lineage["inverse_authorized"] is False
    assert lineage["mpb_replay_authorized"] is False


def test_b4_checkpoint_resume_discovers_latest_epoch_and_preserves_history(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_epoch_40.pt").write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_epoch_160.pt").write_bytes(b"checkpoint")
    (tmp_path / "checkpoint_epoch_bad.pt").write_bytes(b"checkpoint")
    (tmp_path / "history.csv").write_text(
        "epoch,train_data_loss,train_physics_loss,train_normalization_loss,val_data_loss,lambda_pde_active,learning_rate\n"
        "1,1.0,0.0,0.0,0.6,0.0,0.0001\n"
        "2,0.9,0.1,0.0,0.5,0.01,0.00009\n"
        "3,0.8,0.2,0.0,0.55,0.01,0.00008\n",
        encoding="utf-8",
    )
    latest = _latest_epoch_checkpoint(tmp_path)
    assert latest is not None and latest[0] == 160 and latest[1].name == "checkpoint_epoch_160.pt"
    history = _read_history(tmp_path / "history.csv")
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert _history_best_and_stale(history) == pytest.approx((0.5, 1))


def test_b4_dense_k_config_preserves_path_cardinality_and_support() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "research" / "field_eigen_te_b4_dense_k_gate.yaml").read_text(encoding="utf-8"))
    dense = raw["dense_k"]
    assert dense["path_index_count"] == 31 and dense["unique_coordinate_count"] == 30
    assert dense["seen_k_indices"] == [0, 4, 8, 12, 16, 20, 24, 29]
    assert set(dense["resolutions"]) == {64, 96, 128}
    assert raw["admission_contract"] == {"evaluation_split": "test", "required_seeds": [0, 1, 2]}
    assert raw["pino"]["checkpoint_manifest"].endswith("frozen_candidate_manifest.json")
    assert raw["pino"]["checkpoint_manifest"] != "auto"
    assert all(len(case["checkpoint_sha256"]) == 64 for case in raw["fno"]["cases"])
    assert raw["promotion_thresholds"]["residual_calibration"]["quartile_frequency_mre_reduction_min_pct"] == 10.0


def test_b4_validation_selector_dispatches_through_the_b4_runner() -> None:
    from phc2d.forward.field_eigen_b4 import is_field_eigen_b4_dense_config

    root = Path(__file__).resolve().parents[1]
    assert is_field_eigen_b4_dense_config(root / "configs" / "research" / "field_eigen_te_b4_repair_validation_selector.yaml")


def test_b4_solver_floor_gate_keeps_thresholds_and_freezes_real_mpb_calibration() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / "configs" / "research" / "field_eigen_te_b4_dense_k_gate.yaml").read_text(encoding="utf-8"))
    calibrated = yaml.safe_load((root / "configs" / "research" / "field_eigen_te_b4_dense_k_solver_floor_gate.yaml").read_text(encoding="utf-8"))
    assert calibrated["promotion_thresholds"] == original["promotion_thresholds"]
    reference = calibrated["residual_solver_floor_reference"]
    assert reference["status"] == "calibration_from_real_mpb_validation_only"
    assert reference["source_split"] == "val" and reference["source_structure_count"] == 8
    assert reference["audit_decision_sha256"] == "fee4d0628b5acfced46228d5f60ad3812b7981e9aa49dcdf2920bf827d742459"
    assert reference["factors"]["64"] == 1.0
    assert reference["factors"]["96"] > 1.0 and reference["factors"]["128"] > reference["factors"]["96"]


def test_b4_k_path_sidecar_has_duplicate_gamma_and_expected_seen_coordinates() -> None:
    from phc2d.forward.field_eigen_b4 import _read_sidecar

    root = Path(__file__).resolve().parents[1]
    sidecar = _read_sidecar(root / "runs" / "20260628" / "query_dataset_v2_pilot_10000_202606281038" / "k_path_sidecar.json")
    assert sidecar["Nk"] == 31
    assert sidecar["duplicate_coordinate_groups"] == [[0, 30]]
    assert len({tuple(point) for point in sidecar["k_points"]}) == 30
    seen = [0, 4, 8, 12, 16, 20, 24, 29]
    assert all(index < sidecar["Nk"] for index in seen)


def test_b4_lineage_rejects_v2_split_or_hash_mismatch() -> None:
    from phc2d.forward.field_eigen_b4 import _verify_v2_source_lineage

    good = {"test": {"dataset_000001": {"source_hash": "abc"}}}
    metadata = '{"sample_id":"dataset_000001","sample_index":1,"split":"test","rho_hash":"abc"}'
    _verify_v2_source_lineage(good, metadata)
    with pytest.raises(ValueError, match="split mismatch"):
        _verify_v2_source_lineage(good, metadata.replace('"test"', '"val"'))
    with pytest.raises(ValueError, match="rho hash mismatch"):
        _verify_v2_source_lineage(good, metadata.replace('"abc"', '"wrong"'))


def test_b4_manifest_requires_exact_test_split_seed_set_and_hash(tmp_path: Path) -> None:
    from phc2d.forward.field_eigen_b4 import _hash, _resolve_pino_cases

    manifest = tmp_path / "manifest.json"
    payload = {
        "evaluation_split": "test",
        "dataset_hash": "field-hash",
        "model": {"name": "fno_field_v2", "params": {"width": 4}},
        "cases": [
            {"seed": seed, "checkpoint_path": f"checkpoint-{seed}.pt", "checkpoint_sha256": f"{seed}" * 64}
            for seed in (0, 1, 2)
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    raw = {"admission_contract": {"evaluation_split": "test", "required_seeds": [0, 1, 2]}, "pino": {"checkpoint_manifest": str(manifest), "checkpoint_manifest_sha256": _hash(manifest)}}
    cases, lineage = _resolve_pino_cases(raw, expected_dataset_hash="field-hash")
    assert [case["seed"] for case in cases] == [0, 1, 2]
    assert lineage["evaluation_split"] == "test"
    raw["pino"]["checkpoint_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _resolve_pino_cases(raw, expected_dataset_hash="field-hash")
    payload["evaluation_split"] = "val"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    raw["pino"]["checkpoint_manifest_sha256"] = _hash(manifest)
    with pytest.raises(ValueError, match="evaluation_split"):
        _resolve_pino_cases(raw, expected_dataset_hash="field-hash")


def _b4_synthetic_support() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, multiplier in (("fno", 1.0), ("pino", 0.75)):
        for seed in (0, 1, 2):
            for scope in ("seen_k", "unseen_k"):
                rows.append({"kind": kind, "seed": seed, "split": "test", "resolution": 64, "metric_scope": "support", "scope_value": scope, "frequency_mre_pct": 2.0 * multiplier, "e_relative_l2": 0.5 * multiplier, "maxwell_relative_residual_median": 4.0 * multiplier})
    return rows


def test_b4_decision_freezes_thresholds_and_blocks_nonfinite_calibration() -> None:
    from phc2d.forward.field_eigen_b4 import _decision

    thresholds = {
        "frequency": {"mean_fno_factor_max": 1.02, "per_seed_fno_factor_max": 1.05},
        "field": {"seen_k_e_relative_l2_fno_factor_max": 1.02},
        "residual": {"per_seed_fno_factor_max": 0.85, "mean_reduction_min_pct": 20.0},
        "residual_calibration": {"spearman_ci_lower_min_exclusive": 0.0, "quartile_frequency_mre_reduction_min_pct": 10.0},
        "resolution_transfer": {"base_resolution": 64, "evaluation_resolutions": [96, 128], "frequency_mre_degradation_max_pct": 2.0, "residual_median_degradation_max_pct": 20.0},
    }
    calibration = [{"kind": "pino", "seed": seed, "split": "test", "resolution": 64, "all_finite": True, "spearman_bootstrap_ci_lower": 0.1, "quartile_frequency_mre_reduction_pct": 12.0} for seed in (0, 1, 2)]
    transfer = [{"kind": "pino", "seed": seed, "resolution": resolution, "frequency_pass": True, "residual_pass": True, "all_finite": True} for seed in (0, 1, 2) for resolution in (96, 128)]
    decision = _decision(_b4_synthetic_support(), calibration, transfer, thresholds=thresholds, admission_contract={"evaluation_split": "test", "required_seeds": [0, 1, 2]})
    assert decision["thresholds"] == thresholds
    assert decision["physics_frequency_crosscheck_ready"]["status"] == "pass"
    assert decision["physics_residual_rerank_ready"]["status"] == "pass"
    calibration[0]["all_finite"] = False
    calibration[0]["spearman_bootstrap_ci_lower"] = float("nan")
    blocked = _decision(_b4_synthetic_support(), calibration, transfer, thresholds=thresholds, admission_contract={"evaluation_split": "test", "required_seeds": [0, 1, 2]})
    assert blocked["physics_residual_rerank_ready"]["status"] == "blocked"
    assert blocked["inverse_status"] == "blocked"


def test_b4_partial_artifact_resumes_only_matching_complete_input(tmp_path: Path) -> None:
    from phc2d.forward.field_eigen_b4 import _atomic_write_json, _load_resumable_partial, _partial_fingerprint

    partial = tmp_path / "partials" / "pino_seed0__test__r64.json"
    rows = [{"case": "pino_seed0", "frequency_mre_pct": 1.0}, {"case": "pino_seed0", "frequency_mre_pct": 2.0}]
    _atomic_write_json(partial, {"input_fingerprint": "same-input", "rows": rows})
    assert _load_resumable_partial(partial, fingerprint="same-input", expected_rows=2) == rows
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _load_resumable_partial(partial, fingerprint="different-input", expected_rows=2)
    with pytest.raises(ValueError, match="expected 3"):
        _load_resumable_partial(partial, fingerprint="same-input", expected_rows=3)
    common = {
        "input_lineage": {"field_hdf5": "frozen"},
        "case": {"name": "pino_seed0"},
        "identity": {"checkpoint_sha256": "frozen"},
        "split": "test",
        "resolution": 64,
    }
    first = _partial_fingerprint(raw={"output": {"runs_dir": "runs", "resume_run_dir": None}}, **common)
    resumed = _partial_fingerprint(raw={"output": {"runs_dir": "runs", "resume_run_dir": "runs/20260722/existing"}}, **common)
    assert first == resumed


def test_b4_calibration_reports_nonfinite_as_blocked_without_exception() -> None:
    from phc2d.forward.field_eigen_b4 import _calibration_rows

    rows = [
        {"case": "pino_seed0", "kind": "pino", "seed": 0, "split": "test", "resolution": 64, "source_sample_id": f"sample_{index}", "maxwell_relative_residual": float("nan") if index == 0 else float(index), "frequency_mre_pct": float(index + 1)}
        for index in range(4)
    ]
    calibration = _calibration_rows(rows, bootstrap_seed=1, bootstrap_repeats=4)
    assert calibration[0]["calibration_status"] == "blocked_nonfinite"
    assert calibration[0]["all_finite"] is False
