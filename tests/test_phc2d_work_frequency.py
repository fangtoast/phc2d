"""Tests for work-frequency priors, shared gap metrics, and EXP-FWD-012.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.forward.full_cell_metrics import select_largest_gap_edge
from phc2d.forward.work_frequency import (
    bragg_frequency,
    mean_refractive_index,
    nearest_frequency_distance,
    normalized_frequency_weights,
    positive_adjacent_band_spacings,
)
from phc2d.forward.work_frequency_audit import run_work_frequency_audit
from phc2d.runs import validate_run_artifacts


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def test_bragg_prior_uses_material_grid_and_mpb_units() -> None:
    grid = np.stack((np.zeros((3, 5)), np.ones((3, 5))))
    n_bar = mean_refractive_index(grid, epsilon_air=1.0, epsilon_material=9.0)
    assert np.allclose(n_bar, [1.0, 3.0])
    x = bragg_frequency(grid, epsilon_air=1.0, epsilon_material=9.0, direction="X")
    m = bragg_frequency(grid, epsilon_air=1.0, epsilon_material=9.0, direction="M")
    assert np.allclose(x, [0.5, 1.0 / 6.0])
    assert np.allclose(m, np.sqrt(2.0) * x)


@pytest.mark.parametrize("kind", ("inverse_distance", "gaussian"))
def test_frequency_weights_are_finite_and_normalized(kind: str) -> None:
    frequencies = np.asarray(
        [
            [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]],
            [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
        ]
    )
    kwargs = {"sigma": 0.1} if kind == "gaussian" else {}
    weights = normalized_frequency_weights(
        frequencies,
        np.asarray([0.22, 0.44]),
        kind=kind,
        epsilon=0.01,
        **kwargs,
    )
    assert np.isfinite(weights).all()
    assert np.allclose(weights.mean(axis=(1, 2)), 1.0)


def test_frequency_distance_spacing_and_gap_selection_are_shape_driven() -> None:
    frequencies = np.asarray(
        [
            [[0.10, 0.20, 0.50], [0.12, 0.22, 0.52]],
            [[0.11, 0.21, 0.31], [0.13, 0.23, 0.33]],
        ]
    )
    distance = nearest_frequency_distance(frequencies, np.asarray([0.19, 0.30]))
    assert np.allclose(distance, [0.01, 0.01])
    assert np.all(positive_adjacent_band_spacings(frequencies) > 0.0)
    selected = select_largest_gap_edge(
        frequencies[0],
        lower_band_min_one_based=1,
        lower_band_max_one_based=2,
    )
    assert selected.lower_band_one_based == 2
    assert selected.upper_band_one_based == 3
    assert selected.status == "complete_gap"


def _write_audit_lane(path: Path, *, seed: int) -> dict[str, str]:
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(seed)
    train_count = 4002
    val_count = 4
    test_count = 4
    total = train_count + val_count + test_count
    idx_train = np.arange(train_count, dtype=np.int64)
    idx_val = np.arange(train_count, train_count + val_count, dtype=np.int64)
    idx_test = np.arange(train_count + val_count, total, dtype=np.int64)
    grid = rng.uniform(0.2, 0.8, size=(total, 2, 2)).astype(np.float32)
    n_bar = np.mean(np.sqrt(1.0 + 7.9 * grid), axis=(1, 2))
    center = 1.0 / (2.0 * n_bar)
    wh = np.empty((total, 2, 3), dtype=np.float32)
    wh[:, :, 0] = center[:, None] - np.asarray([0.08, 0.07])
    wh[:, :, 1] = center[:, None] - np.asarray([0.02, 0.01])
    wh[:, :, 2] = center[:, None] + np.asarray([0.01, 0.02])
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "phc2d.full_cell_dataset.v2"
        handle.attrs["metadata_json"] = json.dumps(
            {"solver": {"epsilon_air": 1.0, "epsilon_material": 8.9}}
        )
        handle.create_dataset("material_weight_grid_2", data=grid)
        handle.create_dataset("Wh", data=wh)
        handle.create_dataset("k_points", data=np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32))
        handle.create_dataset("idx_train", data=idx_train)
        handle.create_dataset("idx_val", data=idx_val)
        handle.create_dataset("idx_test", data=idx_test)
        handle.create_dataset("sample_id", data=np.asarray([f"sample-{i}".encode() for i in range(total)]))
    return {
        "expected_idx_train_hash": _hash(idx_train),
        "expected_idx_val_hash": _hash(idx_val),
        "expected_idx_test_hash": _hash(idx_test),
    }


def test_exp_fwd_012_audit_writes_no_training_no_test_contract(tmp_path: Path) -> None:
    binary = tmp_path / "binary.h5"
    graded = tmp_path / "graded.h5"
    binary_hashes = _write_audit_lane(binary, seed=1)
    graded_hashes = _write_audit_lane(graded, seed=2)
    config = {
        "experiment": {"experiment_id": "EXP-FWD-012", "stage": "work_frequency_prior_audit"},
        "datasets": {
            "binary": {"path": str(binary), "grid_field": "material_weight_grid_2", **binary_hashes},
            "graded": {"path": str(graded), "grid_field": "material_weight_grid_2", **graded_hashes},
        },
        "audit": {
            "train_size": 4000,
            "train_subset_salt": "exp-fwd-012-test",
            "batch_size": 256,
            "lower_band_min_one_based": 1,
            "lower_band_max_one_based": 2,
            "gates": {"coordinate_coverage_min": 0.85, "gap_hit_improvement_min": 0.05},
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = run_work_frequency_audit(config_path, run_dir=tmp_path / "audit")
    assert validate_run_artifacts(run) == []
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "run.meta.json").read_text(encoding="utf-8"))
    assert results["formal_training_started"] is False
    assert results["test_executed"] is False
    assert results["mpb_meep_used"] is False
    assert meta["test_targets_loaded"] is False
    assert (run / "work_frequency_prior.json").is_file()
    assert (run / "work_frequency_metrics.json").is_file()
