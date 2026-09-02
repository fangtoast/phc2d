"""Contract tests for EXP-PHY-006 full-BZ gap-refinement pilot.

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
from phc2d.physics.bz_gap_refinement import (
    adjacent_gap_records,
    compare_gap_records,
    propose_gap_extrema_refinement_points,
    wrap_reduced_k_points,
)
from phc2d.replay.gap_refinement import (
    run_gap_refinement_pilot,
    run_gap_refinement_study,
)
from phc2d.runs import validate_run_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_periodic_refinement_wraps_deduplicates_and_includes_band_one() -> None:
    points = np.asarray([[-0.49, 0.0], [0.0, 0.0], [0.49, 0.0]])
    bands = np.asarray([[0.20, 0.31, 0.50], [0.10, 0.30, 0.48], [0.19, 0.32, 0.49]])
    proposed = propose_gap_extrema_refinement_points(points, bands, step=0.05)
    assert proposed.shape[1] == 2
    assert len(np.unique(np.round(proposed, 12), axis=0)) == len(proposed)
    assert np.all(proposed >= -0.5) and np.all(proposed < 0.5)
    assert any(np.allclose(point, [-0.46, 0.0]) for point in proposed)
    assert np.allclose(wrap_reduced_k_points(np.asarray([[0.54, -0.55]])), [[-0.46, 0.45]])


def test_gap_comparison_reports_coarse_overestimate_and_status_flip() -> None:
    coarse = adjacent_gap_records(np.asarray([[0.10, 0.20], [0.15, 0.25]]))
    refined = adjacent_gap_records(np.asarray([[0.10, 0.20], [0.23, 0.22]]))
    row = compare_gap_records(coarse, refined)[0]
    assert row["coarse_status"] == "complete_gap"
    assert row["refined_status"] == "pseudo_gap"
    assert row["status_changed"] is True
    assert row["coarse_gap_overestimate"] > 0.0


def _structure(lane: str, index: int) -> FullCellStructureSpec:
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


def _bands(points: np.ndarray) -> np.ndarray:
    radius = np.sum(np.asarray(points, dtype=np.float64) ** 2, axis=1)
    return np.column_stack((0.10 + 0.03 * radius, 0.25 - 0.02 * radius, 0.43 + 0.01 * radius))


def _write_lane(path: Path, lane: str) -> tuple[str, list[dict[str, object]]]:
    h5py = pytest.importorskip("h5py")
    structures = [_structure(lane, index) for index in range(4)]
    axis = np.asarray([-1.0 / 3.0, 0.0, 1.0 / 3.0])
    k_points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    wh = np.stack([_bands(k_points) for _ in structures], axis=0)
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("Wh", data=wh)
        handle.create_dataset("k_points", data=k_points)
        handle.create_dataset("idx_val", data=np.asarray([0, 1], dtype=np.int64))
        handle.create_dataset("idx_test", data=np.asarray([2, 3], dtype=np.int64))
        handle.create_dataset("sample_id", data=[item.sample_id for item in structures], dtype=string)
        handle.create_dataset("geometry_hash", data=[item.geometry_hash for item in structures], dtype=string)
        handle.create_dataset(
            "structure_json",
            data=[json.dumps(item.to_mapping()) for item in structures],
            dtype=string,
        )
    identities = [
        {"dataset_index": index, "geometry_hash": structures[index].geometry_hash}
        for index in (0, 1)
    ]
    return hashlib.sha256(path.read_bytes()).hexdigest(), identities


def _config(tmp_path: Path) -> Path:
    datasets = {}
    for lane in ("binary", "graded"):
        path = tmp_path / f"{lane}.h5"
        digest, samples = _write_lane(path, lane)
        datasets[lane] = {
            "path": str(path),
            "expected_dataset_sha256": digest,
            "samples": samples,
        }
    payload = {
        "experiment": {"experiment_id": "EXP-PHY-006"},
        "datasets": datasets,
        "sampling": {"split": "validation"},
        "mpb": {
            "polarization": "TE",
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "resolution": 64,
            "verification_resolution": 96,
            "mesh_size": 3,
            "eigensolver_tolerance": 1.0e-7,
            "band_guard_count": 4,
            "band_order_policy": "frequency_ascending",
        },
        "refinement": {
            "initial_spacing": 1.0 / 3.0,
            "levels": 2,
            "lower_band_min_one_based": 1,
            "lower_band_max_one_based": 2,
        },
        "gates": {
            "last_refinement_gap_change_max": 5.0e-4,
            "resolution_edge_change_max": 5.0e-4,
            "coarse_gap_change_material": 5.0e-4,
        },
        "execution": {"mode": "pilot_real_mpb", "formal_solver_authorized": True},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_gap_refinement_pilot_is_validation_only_and_writes_complete_artifacts(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def solver(structure: object, *, k_points: np.ndarray, config: object):
        return _bands(k_points), {"backend": "synthetic", "resolution": config.resolution}

    run = run_gap_refinement_pilot(config, run_dir=tmp_path / "pilot", solver=solver)
    assert validate_run_artifacts(run) == []
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    rows = list(__import__("csv").DictReader((run / "gap_refinement_samples.csv").open(encoding="utf-8")))
    manifest = [json.loads(line) for line in (run / "sample_manifest.jsonl").read_text().splitlines()]
    assert results["pilot_real_mpb_completed"] is True
    assert results["test_targets_loaded"] is False
    assert len(rows) == 8
    assert {row["split"] for row in manifest} == {"validation"}
    assert {int(row["dataset_index"]) for row in manifest} == {0, 1}


def test_gap_refinement_cli_help_is_lazy() -> None:
    code = """
import sys
from phc2d.cli import main
try:
    main(['replay', 'gap-refinement', '--help'])
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


def test_adaptive_study_selects_frozen_validation_manifest_and_unlocks_on_stability(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    manifest_rows = []
    for lane in ("binary", "graded"):
        for identity in payload["datasets"][lane].pop("samples"):
            manifest_rows.append({"lane": lane, "split": "validation", **identity})
    manifest = tmp_path / "source_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    payload["experiment"]["experiment_id"] = "EXP-PHY-007"
    payload["sampling"].update(
        {
            "source_manifest": str(manifest),
            "expected_source_manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "sample_count_per_lane": 2,
        }
    )
    payload["refinement"] = {
        "initial_spacing": 1.0 / 3.0,
        "min_levels": 1,
        "max_levels": 2,
        "required_consecutive_stable_levels": 1,
        "lower_band_min_one_based": 1,
        "lower_band_max_one_based": 2,
    }
    payload["gates"].update(
        {
            "last_refinement_edge_change_max": 1.0,
            "resolution_edge_change_max": 1.0,
        }
    )
    payload["execution"] = {
        "mode": "adaptive_real_mpb",
        "formal_solver_authorized": True,
    }
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def solver(structure: object, *, k_points: np.ndarray, config: object):
        return _bands(k_points), {"backend": "synthetic", "resolution": config.resolution}

    run = run_gap_refinement_study(
        config,
        run_dir=tmp_path / "adaptive",
        solver=solver,
    )
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert results["status"] == "adaptive_label_gate_pass"
    assert results["loss_screen_unlocked"] is True
    assert results["formal_physical_validation_completed"] is True
    assert metrics["sample_count"] == 4
