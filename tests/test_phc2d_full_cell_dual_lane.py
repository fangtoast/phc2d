"""Tests for balanced dual-lane Full-Wh training and upgrade contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.forward.full_cell import _validate_and_resolve as resolve_single
from phc2d.forward.full_cell_dual_lane import run_full_cell_dual_lane
from phc2d.runs import validate_run_artifacts


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _write_lane(path: Path, *, seed: int) -> dict[str, str]:
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(seed)
    idx_train = np.arange(6, dtype=np.int64)
    idx_val = np.arange(6, 8, dtype=np.int64)
    idx_test = np.arange(8, 10, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "phc2d.full_cell.v1"
        handle.create_dataset("material_weight_grid_8", data=rng.random((10, 8, 8), dtype=np.float32))
        handle.create_dataset("Wh", data=0.2 + rng.random((10, 3, 2), dtype=np.float32))
        handle.create_dataset("k_points", data=np.asarray([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]], dtype=np.float32))
        handle.create_dataset("idx_train", data=idx_train)
        handle.create_dataset("idx_val", data=idx_val)
        handle.create_dataset("idx_test", data=idx_test)
    return {
        "expected_idx_train_hash": _hash(idx_train),
        "expected_idx_val_hash": _hash(idx_val),
        "expected_idx_test_hash": _hash(idx_test),
    }


def test_single_lane_upgrade_head_override_is_resolved() -> None:
    config = {
        "experiment": {"experiment_id": "head_override", "lane": "binary"},
        "dataset": {"path": "data.h5"},
        "representation": {"sampling": "continuous", "threshold": None, "resolution": 64},
        "models": {"tfno2d_band_v3": {"width": 8}},
        "training": {"learning_rates": {"tfno2d_band_v3": 5.0e-4}},
        "runtime": {},
        "output": {},
    }
    resolved = resolve_single(
        config,
        model_name="tfno2d_band_v3",
        seed=0,
        admission_probe=True,
        head_kind_override="k_conditioned",
    )
    assert resolved["model"]["params"]["head_kind"] == "k_conditioned"


def test_dual_lane_cpu_admission_writes_lane_specific_contract(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    binary = tmp_path / "binary.h5"
    graded = tmp_path / "graded.h5"
    binary_hashes = _write_lane(binary, seed=21)
    graded_hashes = _write_lane(graded, seed=22)
    config = {
        "experiment": {"experiment_id": "tiny_dual", "lane": "joint"},
        "datasets": {
            "binary": {"path": str(binary), "grid_field": "material_weight_grid_8", **binary_hashes},
            "graded": {"path": str(graded), "grid_field": "material_weight_grid_8", **graded_hashes},
        },
        "representation": {
            "sampling": "continuous",
            "threshold": None,
            "resolution": 8,
            "physical_equivalence": True,
        },
        "model": {
            "name": "dual_lane_band_operator_v1",
            "params": {
                "width": 8,
                "depth": 1,
                "modes": [2, 2],
                "factorization_rank": 0.5,
                "norm_groups": 2,
                "head_hidden_dim": 8,
                "band_embedding_dim": 4,
                "fourier_frequencies": [1, 2],
            },
        },
        "training": {
            "max_epochs": 100,
            "early_stopping_patience": 15,
            "optimizer": "adamw",
            "learning_rate": 5.0e-4,
            "weight_decay": 1.0e-5,
            "micro_batch_size": 4,
            "effective_batch_size": 4,
            "loss": {
                "name": "equal_lane_gap_aware_v1",
                "lower_band_min_one_based": 1,
                "lower_band_max_one_based": 1,
                "full_spectrum_weight": 1.0,
                "gap_profile_weight": 0.25,
                "selected_edge_weight": 0.25,
                "selected_width_weight": 0.25,
                "selected_center_weight": 0.25,
            },
        },
        "runtime": {
            "device": "cpu",
            "require_accelerator": False,
            "precision": "fp32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "num_workers": 0,
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path = tmp_path / "dual.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = run_full_cell_dual_lane(
        config_path,
        seed=0,
        head_kind="k_conditioned",
        run_dir=tmp_path / "probe",
        admission_probe=True,
    )
    assert validate_run_artifacts(run_dir) == []
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "run.meta.json").read_text(encoding="utf-8"))
    assert results["status"] == "dual_lane_admission_probe_pass"
    assert results["test_executed"] is False
    assert metrics["binary_n_train"] == 6
    assert metrics["graded_n_train"] == 6
    assert metrics["loss_contract"]["name"] == "equal_lane_gap_aware_v1"
    assert set(metrics["gap_validation"]) == {"binary", "graded"}
    assert "joint_band_k_accuracy" in metrics["gap_validation"]["binary"]
    history = (run_dir / "history.csv").read_text(encoding="utf-8")
    assert "train_gap_profile_loss" in history
    assert set(meta["dataset_sha256"]) == {"binary", "graded"}
    assert meta["k_points_sha256"]


def test_frequency_weighted_admission_does_not_read_test_targets(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    h5py = pytest.importorskip("h5py")
    binary = tmp_path / "binary.h5"
    graded = tmp_path / "graded.h5"
    binary_hashes = _write_lane(binary, seed=31)
    graded_hashes = _write_lane(graded, seed=32)
    for path in (binary, graded):
        with h5py.File(path, "r+") as handle:
            handle["Wh"][8:10] = np.nan
    config = {
        "experiment": {"experiment_id": "EXP-FWD-018", "lane": "joint"},
        "datasets": {
            "binary": {"path": str(binary), "grid_field": "material_weight_grid_8", **binary_hashes},
            "graded": {"path": str(graded), "grid_field": "material_weight_grid_8", **graded_hashes},
        },
        "representation": {"sampling": "continuous", "threshold": None, "resolution": 8, "physical_equivalence": True},
        "model": {"name": "dual_lane_band_operator_v1", "params": {
            "width": 8, "depth": 1, "modes": [2, 2], "factorization_rank": 0.5,
            "norm_groups": 2, "head_hidden_dim": 8, "band_embedding_dim": 4, "fourier_frequencies": [1, 2],
        }},
        "training": {
            "max_epochs": 100, "early_stopping_patience": 15, "optimizer": "adamw", "learning_rate": 5.0e-4,
            "weight_decay": 1.0e-5, "micro_batch_size": 4, "effective_batch_size": 4,
            "loss": {"name": "equal_lane_bounded_inverse_frequency_v1", "epsilon_air": 1.0,
                     "epsilon_material": 8.9, "epsilon_f": 0.01, "rho": 4.0, "alpha": 0.25},
        },
        "runtime": {"device": "cpu", "require_accelerator": False, "precision": "fp32", "amp_enabled": False,
                    "tf32_enabled": False, "num_workers": 0},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path = tmp_path / "frequency.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = run_full_cell_dual_lane(config_path, seed=0, head_kind="dense", run_dir=tmp_path / "probe", admission_probe=True)
    assert validate_run_artifacts(run_dir) == []
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["test_executed"] is False
    assert (run_dir / "frequency_center_manifest.jsonl").is_file()
    assert (run_dir / "window_metrics.json").is_file()
