"""Tests for the evidence boundary of full-cell dataset generation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.cli import build_parser
from phc2d.data.full_cell import FullCellStructureSpec, load_full_cell_hdf5
from phc2d.data.full_cell_generation import run_full_cell_generation, write_combined_full_cell_manifest


def deterministic_fake_solver(structure, *, k_points, config):
    assert isinstance(structure, FullCellStructureSpec)
    base = 0.1 + (int(structure.geometry_hash[:4], 16) % 100) / 10000.0
    bands = base + np.arange(int(config.num_bands), dtype=np.float32)[None, :] * 0.05
    bands = np.repeat(bands, len(k_points), axis=0)
    return bands, {"solver_backend": "deterministic_fake"}


def failing_full_cell_solver(structure, *, k_points, config):
    raise RuntimeError(f"intentional failure for {structure.sample_id}")


def _write_config(tmp_path: Path) -> Path:
    payload = {
        "experiment": {
            "experiment_id": "full_cell_unit",
            "family": "research",
            "stage": "dataset_generation",
        },
        "dataset": {
            "protocol": "full_cell_v1",
            "lane": "graded_medium",
            "output_path": str(tmp_path / "dataset.h5"),
            "sample_count": 4,
            "split": {"train": 2, "val": 1, "test": 1},
            "seed": 17,
            "raster_resolutions": [32, 64, 128],
        },
        "generator": {
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "audit_resolution": 64,
            "fourier_cutoff": 2,
            "max_attempts": 128,
        },
        "solver": {
            "backend": "mpb",
            "environment": "wsl_meep_env",
            "use_proxy": False,
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "resolution": 64,
            "mesh_size": 3,
            "polarization": "TE",
            "num_bands": 10,
            "band_guard_count": 4,
            "band_order_policy": "frequency_ascending",
            "eigensolver_tolerance": 1.0e-7,
        },
        "k_grid": {
            "family": "square_full_bz",
            "n_side": 9,
            "coordinate_system": "mpb_reduced_reciprocal_coordinates",
            "endpoint_policy": "centered_unique_no_boundary_duplicates",
        },
        "workers": {
            "workers": 2,
            "min_workers": 1,
            "max_workers": 2,
            "worker_backend": "thread",
            "fail_fast": False,
            "suppress_solver_output": True,
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_injected_solver_writes_complete_but_unaccepted_smoke_run(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    config = _write_config(tmp_path)
    run_dir = tmp_path / "run"
    result = run_full_cell_generation(
        config,
        run_dir=run_dir,
        skip_environment_check=True,
        solver_fn=deterministic_fake_solver,
    )

    assert result == run_dir
    for name in (
        "run.log",
        "run.meta.json",
        "config.resolved.yaml",
        "results.json",
        "metrics.json",
        "history.csv",
        "dataset_manifest.yaml",
        "solver_provenance.json",
        "sample_metadata.jsonl",
        "failed_case_ledger.jsonl",
    ):
        assert (run_dir / name).is_file(), name
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    dataset = load_full_cell_hdf5(tmp_path / "dataset.h5")

    assert results["status"] == "test_solver_smoke_only"
    assert provenance["acceptance_eligible"] is False
    assert dataset.Wh.shape == (4, 81, 10)
    assert sorted(np.concatenate([dataset.idx_train, dataset.idx_val, dataset.idx_test]).tolist()) == [0, 1, 2, 3]


def test_solver_failure_is_preserved_and_dataset_is_not_written(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    run_dir = tmp_path / "failed_run"
    run_full_cell_generation(
        config,
        run_dir=run_dir,
        skip_environment_check=True,
        solver_fn=failing_full_cell_solver,
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    ledger = (run_dir / "failed_case_ledger.jsonl").read_text(encoding="utf-8")
    assert results["status"] == "solver_execution_failed"
    assert results["dataset_written"] is False
    assert "intentional failure" in ledger
    assert not (tmp_path / "dataset.h5").exists()


def test_data_generate_cli_parser_does_not_import_meep() -> None:
    before = set(sys.modules)
    parser = build_parser()
    namespace = parser.parse_args(["data", "generate", "--config", "pilot.yaml", "--workers", "auto"])

    assert namespace.command == "data generate"
    assert namespace.workers == "auto"
    assert "meep" not in set(sys.modules) - before


def test_combined_manifest_refuses_promotion_without_convergence(tmp_path: Path) -> None:
    lane = {
        "status": "pilot_generated_pending_convergence",
        "dataset_path": "data/lane.h5",
        "dataset_sha256": "abc",
    }
    graded = tmp_path / "graded.yaml"
    binary = tmp_path / "binary.yaml"
    graded.write_text(yaml.safe_dump(lane), encoding="utf-8")
    binary.write_text(yaml.safe_dump(lane), encoding="utf-8")

    with pytest.raises(ValueError, match="resolution_convergence_pass"):
        write_combined_full_cell_manifest(
            tmp_path / "combined.yaml",
            graded_manifest=graded,
            binary_manifest=binary,
            status="pilot_pass",
        )

    convergence = tmp_path / "convergence.json"
    convergence.write_text(json.dumps({"status": "resolution_convergence_pass"}), encoding="utf-8")
    result = write_combined_full_cell_manifest(
        tmp_path / "combined.yaml",
        graded_manifest=graded,
        binary_manifest=binary,
        convergence_evidence=convergence,
        status="pilot_pass",
    )
    payload = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert payload["status"] == "pilot_pass"
    assert payload["merge_arrays"] is False
    assert payload["resolution_convergence"]["status"] == "resolution_convergence_pass"
