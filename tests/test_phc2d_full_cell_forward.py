"""Tests for formal full-cell forward training and admission probes.

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

from phc2d.forward.full_cell import (
    _nested_train_indices,
    _validate_and_resolve,
    run_full_cell_forward,
)
from phc2d.runs import validate_run_artifacts


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def test_nested_train_indices_are_deterministic_and_nested() -> None:
    indices = np.arange(40, dtype=np.int64)
    sample_ids = [f"sample_{index:03d}" for index in indices]
    selected_8 = _nested_train_indices(indices, sample_ids, train_size=8, salt="frozen")
    selected_20 = _nested_train_indices(indices, sample_ids, train_size=20, salt="frozen")
    selected_40 = _nested_train_indices(indices, sample_ids, train_size=40, salt="frozen")

    assert set(selected_8).issubset(set(selected_20))
    assert set(selected_20).issubset(set(selected_40))
    assert np.array_equal(
        selected_20,
        _nested_train_indices(indices, sample_ids, train_size=20, salt="frozen"),
    )


def test_formal_50k_overrides_accept_independent_epoch_and_train_size() -> None:
    config = {
        "experiment": {"experiment_id": "full_cell_50k", "lane": "binary"},
        "dataset": {"path": "data.h5"},
        "representation": {
            "sampling": "continuous",
            "threshold": None,
            "resolution": 64,
            "physical_equivalence": True,
        },
        "models": {"mlp": {}},
        "training": {
            "max_epochs": 20,
            "early_stopping_patience": 0,
            "optimizer": "adamw",
            "effective_batch_size": 64,
            "learning_rates": {"mlp": 3.0e-4},
        },
        "runtime": {"precision": "fp32", "amp_enabled": False, "tf32_enabled": False},
        "output": {},
    }

    resolved = _validate_and_resolve(
        config,
        model_name="mlp",
        seed=0,
        admission_probe=False,
        train_size_override=8000,
        max_epochs_override=200,
    )

    assert resolved["training"]["train_size"] == 8000
    assert resolved["training"]["max_epochs"] == 200
    assert resolved["experiment"]["experiment_id"].endswith("_n8000_e200")


def test_full_cell_forward_cpu_admission_probe_writes_contract(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional")
    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "tiny.h5"
    rng = np.random.default_rng(12)
    idx_train = np.arange(6, dtype=np.int64)
    idx_val = np.arange(6, 8, dtype=np.int64)
    idx_test = np.arange(8, 10, dtype=np.int64)
    with h5py.File(dataset, "w") as handle:
        handle.attrs["schema_version"] = "phc2d.full_cell.v1"
        handle.create_dataset("material_weight_grid_8", data=rng.random((10, 8, 8), dtype=np.float32))
        handle.create_dataset("Wh", data=0.2 + rng.random((10, 3, 2), dtype=np.float32))
        handle.create_dataset("k_points", data=rng.random((3, 2), dtype=np.float32))
        handle.create_dataset("idx_train", data=idx_train)
        handle.create_dataset("idx_val", data=idx_val)
        handle.create_dataset("idx_test", data=idx_test)
    config = {
        "experiment": {"experiment_id": "tiny_full_cell", "lane": "graded"},
        "dataset": {
            "path": str(dataset),
            "grid_field": "material_weight_grid_8",
            "expected_idx_train_hash": _hash(idx_train),
            "expected_idx_val_hash": _hash(idx_val),
            "expected_idx_test_hash": _hash(idx_test),
        },
        "representation": {
            "sampling": "deterministic_threshold",
            "threshold": 0.5,
            "resolution": 8,
            "physical_equivalence": False,
        },
        "models": {
            "mlp": {"hidden_dim": 8, "depth": 1},
            "res_mlp_wide": {"hidden_dim": 8, "depth": 1},
            "lifted_cnn": {"channels": 4},
            "fno2d_band_v2": {"width": 4, "depth": 1, "modes": [2, 2], "band_modes": [2, 2], "fourier_rank": 2},
            "res_mlp_gated_v2": {
                "hidden_dim": 8,
                "depth": 1,
                "head_kind": "k_conditioned",
                "head_hidden_dim": 8,
                "band_embedding_dim": 4,
                "fourier_frequencies": [1, 2],
            },
        },
        "training": {
            "max_epochs": 100,
            "early_stopping_patience": 15,
            "optimizer": "adamw",
            "weight_decay": 1.0e-5,
            "learning_rates": {"mlp": 3.0e-4, "res_mlp_wide": 3.0e-4, "lifted_cnn": 1.0e-3, "fno2d_band_v2": 1.0e-3, "res_mlp_gated_v2": 3.0e-4},
            "micro_batch_size": 2,
            "effective_batch_size": 2,
        },
        "runtime": {"device": "cpu", "require_accelerator": False, "precision": "fp32", "amp_enabled": False, "tf32_enabled": False, "num_workers": 0},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = run_full_cell_forward(
        config_path,
        model_name="mlp",
        seed=0,
        run_dir=tmp_path / "probe",
        admission_probe=True,
    )
    assert validate_run_artifacts(run_dir) == []
    assert (run_dir / "figures" / "training_curve.png").is_file()
    assert (run_dir / "figures" / "training_curve.pdf").is_file()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["status"] == "full_cell_forward_admission_probe_pass"
    assert results["formal_training_started"] is False
    assert results["test_executed"] is False

    final_fit_dir = run_full_cell_forward(
        config_path,
        model_name="mlp",
        seed=0,
        run_dir=tmp_path / "probe_final_fit",
        admission_probe=True,
        train_size=10,
    )
    final_fit_results = json.loads(
        (final_fit_dir / "results.json").read_text(encoding="utf-8")
    )
    final_fit_metrics = json.loads(
        (final_fit_dir / "metrics.json").read_text(encoding="utf-8")
    )
    final_fit_meta = json.loads(
        (final_fit_dir / "run.meta.json").read_text(encoding="utf-8")
    )
    assert final_fit_results["selection_split"] == "none_final_fit"
    assert final_fit_results["checkpoint_role"] == "final_epoch_final_fit"
    assert final_fit_results["validation_metrics_admissible"] is False
    assert final_fit_metrics["best_validation_bands_2_10_mre_pct"] is None
    assert final_fit_metrics["monitor_is_unbiased_validation"] is False
    assert final_fit_meta["validation_overlap_count"] == 2
    assert final_fit_meta["test_overlap_count"] == 2

    query_dir = run_full_cell_forward(
        config_path,
        model_name="res_mlp_gated_v2",
        seed=0,
        run_dir=tmp_path / "probe_query_head",
        admission_probe=True,
    )
    query_meta = json.loads((query_dir / "run.meta.json").read_text(encoding="utf-8"))
    assert query_meta["k_points_sha256"]


def test_threshold_proxy_requires_false_physical_equivalence(tmp_path: Path) -> None:
    config = {
        "experiment": {"experiment_id": "bad", "lane": "graded"},
        "dataset": {"path": "missing.h5"},
        "representation": {"sampling": "deterministic_threshold", "threshold": 0.5, "physical_equivalence": True},
        "models": {"mlp": {}},
        "training": {"learning_rates": {"mlp": 3.0e-4}},
        "runtime": {},
        "output": {},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="physical_equivalence"):
        run_full_cell_forward(path, model_name="mlp", seed=0, run_dir=tmp_path / "run", admission_probe=True)


def test_resolution_override_rejects_unplanned_grid(tmp_path: Path) -> None:
    config = {
        "experiment": {"experiment_id": "bad_resolution", "lane": "binary"},
        "dataset": {"path": "missing.h5"},
        "representation": {"sampling": "continuous", "threshold": None, "resolution": 64, "physical_equivalence": True},
        "models": {"mlp": {}},
        "training": {"learning_rates": {"mlp": 3.0e-4}},
        "runtime": {},
        "output": {},
    }
    path = tmp_path / "bad_resolution.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="32, 64, or 128"):
        run_full_cell_forward(
            path,
            model_name="mlp",
            seed=0,
            run_dir=tmp_path / "run_resolution",
            admission_probe=True,
            resolution=96,
        )
