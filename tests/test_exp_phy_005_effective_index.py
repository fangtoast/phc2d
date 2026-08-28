"""Contract tests for EXP-PHY-005 effective-index physical validation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.data.full_cell import FullCellStructureSpec
from phc2d.forward.full_cell_metrics import enumerate_adjacent_gaps
from phc2d.forward.work_frequency import bragg_frequency, mean_refractive_index
from phc2d.physics.effective_index_validation import (
    build_near_gamma_k_points,
    fit_gamma_slope,
)
from phc2d.physics.effective_medium import (
    effective_refractive_index,
    folded_square_lattice_frequencies,
)
from phc2d.replay.effective_index import run_effective_index_audit
from phc2d.runs import validate_run_artifacts


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("epsilon", (1.0, 4.0, 8.9))
def test_effective_index_homogeneous_te_tm(epsilon: float) -> None:
    grid = np.full((7, 11), epsilon)
    expected = np.sqrt(epsilon)
    assert effective_refractive_index(grid, "TE") == pytest.approx(expected)
    assert effective_refractive_index(grid, "TM") == pytest.approx(expected)


def test_effective_index_candidates_are_distinct_for_nonuniform_medium() -> None:
    epsilon = np.asarray([[1.0, 1.0], [9.0, 9.0]])
    old = float(np.mean(np.sqrt(epsilon)))
    te = effective_refractive_index(epsilon, "TE")
    tm = effective_refractive_index(epsilon, "TM")
    assert te != pytest.approx(old)
    assert tm != pytest.approx(old)
    assert te < old < tm


@pytest.mark.parametrize(
    ("values", "polarization"),
    [([[1.0, 0.0]], "TE"), ([[1.0, np.nan]], "TM"), ([[1.0]], "XX")],
)
def test_effective_index_rejects_invalid_inputs(values: list[list[float]], polarization: str) -> None:
    with pytest.raises(ValueError):
        effective_refractive_index(np.asarray(values), polarization)


def test_folded_square_lattice_frequencies_are_independently_named() -> None:
    values = folded_square_lattice_frequencies(2.0)
    assert values.gamma_g1 == pytest.approx(0.5)
    assert values.x == pytest.approx(0.25)
    assert values.m == pytest.approx(np.sqrt(2.0) / 4.0)


@pytest.mark.parametrize("band_count", (3, 10, 12))
def test_adjacent_gap_enumeration_is_shape_driven_and_includes_band_one(
    band_count: int,
) -> None:
    base = np.arange(band_count, dtype=np.float64)[None, :] * 0.2
    spectra = base + np.asarray([[0.00], [0.02], [0.01]])
    gaps = enumerate_adjacent_gaps(spectra)
    assert len(gaps) == band_count - 1
    assert (gaps[0].lower_band_one_based, gaps[0].upper_band_one_based) == (1, 2)
    assert gaps[0].lower_edge_k_index == 1
    assert gaps[0].upper_edge_k_index == 0


def test_adjacent_gap_enumeration_marks_pseudo_gap() -> None:
    spectra = np.asarray([[0.10, 0.15], [0.20, 0.18]])
    gap = enumerate_adjacent_gaps(spectra)[0]
    assert gap.status == "pseudo_gap"
    assert gap.signed_width < 0.0


def test_legacy_work_frequency_contract_is_unchanged() -> None:
    grid = np.asarray([[[0.0, 1.0], [0.0, 1.0]]])
    old_index = mean_refractive_index(grid, epsilon_air=1.0, epsilon_material=9.0)
    assert old_index.tolist() == [2.0]
    assert bragg_frequency(
        grid, epsilon_air=1.0, epsilon_material=9.0, direction="X"
    ).tolist() == [0.25]


def test_near_gamma_template_has_nine_unique_points_and_exact_slope() -> None:
    kappa = np.asarray([0.0, 1 / 256, 1 / 128, 1 / 64, 1 / 32])
    points, directions = build_near_gamma_k_points(kappa)
    assert points.shape == (9, 2)
    assert len(np.unique(points, axis=0)) == 9
    assert np.allclose(np.linalg.norm(points[directions["x_indices"]], axis=1), kappa)
    assert np.allclose(np.linalg.norm(points[directions["m_indices"]], axis=1), kappa)
    fit = fit_gamma_slope(kappa, 0.07 + kappa / 2.5)
    assert fit.effective_index == pytest.approx(2.5)
    assert fit.r2_origin == pytest.approx(1.0)
    assert fit.window_drift == pytest.approx(0.0)


def _structure(index: int, lane: str) -> FullCellStructureSpec:
    return FullCellStructureSpec(
        sample_id=f"{lane}-{index}",
        generator_family="binary_spectral_threshold",
        material_kind="binary" if lane == "binary" else "continuous",
        parameters={"modes": [], "threshold": float(index)},
        parameter_names=("threshold",),
        parameter_vector=(float(index),),
        parameter_mask=(True,),
        metadata={"lane": lane},
    )


def _write_lane(path: Path, lane: str) -> str:
    h5py = pytest.importorskip("h5py")
    structures = [_structure(index, lane) for index in range(8)]
    raster = np.linspace(0.15, 0.85, 8 * 4 * 4, dtype=np.float32).reshape(8, 4, 4)
    wh = np.empty((8, 3, 4), dtype=np.float32)
    for index in range(8):
        wh[index] = np.asarray(
            [[0.05, 0.18, 0.32, 0.48], [0.08, 0.20, 0.34, 0.50], [0.06, 0.19, 0.33, 0.49]]
        )
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["metadata_json"] = json.dumps(
            {
                "solver": {
                    "polarization": "TE",
                    "epsilon_air": 1.0,
                    "epsilon_material": 8.9,
                    "resolution": 64,
                    "mesh_size": 3,
                    "eigensolver_tolerance": 1.0e-7,
                }
            }
        )
        handle.create_dataset("Wh", data=wh)
        handle.create_dataset("idx_train", data=np.arange(4, dtype=np.int64))
        handle.create_dataset("idx_val", data=np.arange(4, 6, dtype=np.int64))
        handle.create_dataset("idx_test", data=np.arange(6, 8, dtype=np.int64))
        handle.create_dataset("sample_id", data=[item.sample_id for item in structures], dtype=string)
        handle.create_dataset("family_id", data=[item.generator_family for item in structures], dtype=string)
        handle.create_dataset("material_kind", data=[item.material_kind for item in structures], dtype=string)
        handle.create_dataset("geometry_hash", data=[item.geometry_hash for item in structures], dtype=string)
        handle.create_dataset("structure_json", data=[json.dumps(item.to_mapping()) for item in structures], dtype=string)
        handle.create_dataset("content_hash_r4", data=[f"content-{index}" for index in range(8)], dtype=string)
        handle.create_dataset("material_weight_grid_4", data=raster)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _smoke_config(tmp_path: Path) -> Path:
    datasets = {}
    for lane in ("binary", "graded"):
        path = tmp_path / f"{lane}.h5"
        digest = _write_lane(path, lane)
        datasets[lane] = {
            "path": str(path),
            "expected_dataset_sha256": digest,
            "raster_field": "material_weight_grid_4",
            "content_hash_field": "content_hash_r4",
        }
    config = {
        "experiment": {"experiment_id": "EXP-PHY-005"},
        "datasets": datasets,
        "sampling": {"slope_train_per_lane": 1, "slope_validation_per_lane": 1},
        "mpb": {
            "polarization": "TE", "epsilon_air": 1.0, "epsilon_material": 8.9,
            "resolution": 64, "mesh_size": 3, "eigensolver_tolerance": 1.0e-7,
            "num_bands": 2, "band_guard_count": 4,
            "band_order_policy": "frequency_ascending",
            "kappa": [0.0, 1 / 256, 1 / 128, 1 / 64, 1 / 32],
        },
        "audit": {
            "gap_train_size_per_lane": 2, "gap_validation_size_per_lane": 2,
            "train_subset_salt": "synthetic-exp-phy-005", "lower_band_min_one_based": 1,
        },
        "gates": {
            "sentinel_relative_error_max": 0.005,
            "sentinel_direction_difference_max": 0.005,
            "slope_r2_min": 0.999, "slope_r2_pass_rate_min": 0.95,
            "slope_window_drift_max": 0.02, "slope_window_drift_pass_rate_min": 0.90,
            "anisotropy_median_max": 0.02, "anisotropy_p90_max": 0.05,
            "oracle_pair_agreement_min": 0.80, "candidate_out_of_range_rate_max": 0.05,
            "train_validation_tv_max": 0.10, "complete_subset_min": 30,
            "complete_pair_agreement_min": 0.75,
        },
        "execution": {"mode": "engineering_smoke"},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_synthetic_hdf5_preflight_never_loads_test_targets(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    run = run_effective_index_audit(config, run_dir=tmp_path / "preflight", preflight=True)
    assert validate_run_artifacts(run) == []
    meta = json.loads((run / "run.meta.json").read_text(encoding="utf-8"))
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in (run / "sample_manifest.jsonl").read_text().splitlines()]
    assert meta["test_targets_loaded"] is False
    assert results["formal_physical_validation_completed"] is False
    assert {row["split"] for row in manifest} == {"train", "validation"}
    assert all(row["dataset_index"] < 6 for row in manifest)


def test_solver_failures_are_recorded_per_structure(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)

    def uniform_solver(epsilon: float, *, k_points: np.ndarray, config: object):
        radius = np.linalg.norm(k_points, axis=1)
        bands = np.column_stack((radius / np.sqrt(epsilon), 0.5 + radius))
        return bands, {"solver": "synthetic-uniform"}

    def failing_structure_solver(*args: object, **kwargs: object):
        raise RuntimeError("synthetic structure failure")

    run = run_effective_index_audit(
        config,
        run_dir=tmp_path / "blocked",
        structure_solver=failing_structure_solver,
        uniform_solver=uniform_solver,
    )
    failures = [
        json.loads(line)
        for line in (run / "solver_failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["status"] == "blocked"
    assert len(failures) == 4
    assert {(row["lane"], row["split"]) for row in failures} == {
        ("binary", "train"), ("binary", "validation"),
        ("graded", "train"), ("graded", "validation"),
    }
    assert validate_run_artifacts(run) == []


def test_effective_index_cli_help_is_lazy() -> None:
    code = """
import sys
from phc2d.cli import main
try:
    main(['replay', 'effective-index', '--help'])
except SystemExit as exc:
    assert exc.code == 0
print(sorted(set(sys.modules) & {'h5py', 'meep', 'torch'}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.rstrip().endswith("[]")
