"""Tests for Query Dataset V2 pilot generation provenance.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from scripts.run_query_dataset_v2_pilot_generation import (
    REQUIRED_SOLVER_PROVENANCE_FIELDS,
    derive_grid_shape,
    derive_n_wedge,
    load_query_pilot_config,
    run_generation,
)


def _base_config(tmp_path: Path, *, tolerance: float | None = 1.0e-6, use_proxy: bool = False) -> dict:
    return {
        "experiment_id": "query_dataset_v2_pilot_unit",
        "dataset": {
            "output_path": str(tmp_path / "query_dataset_v2_pilot_unit.npz"),
            "sample_count": 3,
            "split": {"train": 1, "val": 1, "test": 1},
            "sample_id_policy": "stable_generated_id",
            "seed": 7,
            "structure_generator": {"name": "legacy_random", "prob_high": 0.5},
        },
        "geometry": {
            "representation": "rho_wedge",
            "grid_shape_source": "config",
            "grid_shape": [4, 4],
            "n_wedge": "from_grid_shape",
            "lifting": "p4m_wedge_binary_pixels_as_mpb_blocks_air_background",
        },
        "solver": {
            "backend": "mpb",
            "environment": "wsl_meep_env",
            "use_proxy": use_proxy,
            "lattice": "square",
            "lattice_constant": 1.0,
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "resolution": 8,
            "polarization": "TE",
            "num_bands": 2,
            "band_guard_count": 1,
            "band_order_policy": "frequency_ascending",
            "eigensolver_tolerance": tolerance,
            "frequency_normalization": "mpb_dimensionless_a_over_lambda",
        },
        "k_path": {
            "source": "generated_from_KPathSpec",
            "family": "square_gxm",
            "k_path_id": "query_dataset_v2_unit_square_gxm",
            "n_total": 4,
            "coordinate_system": "mpb_reduced_reciprocal_coordinates",
            "endpoint_policy": "explicit",
        },
        "output": {
            "runs_dir": str(tmp_path / "runs"),
            "write_manifest": True,
            "write_k_path_sidecar": True,
            "write_solver_provenance": True,
            "write_metadata_jsonl": True,
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "query_dataset_v2_pilot_unit.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _fake_solver(rho, *, k_points, config):
    fill = float(np.asarray(rho, dtype=np.float32).sum())
    bands = np.full((len(k_points), int(config.num_bands)), fill, dtype=np.float32)
    return bands, {
        "mpb_backend": "fake_unit_solver",
        "num_bands": int(config.num_bands),
        "band_guard_count": int(config.band_guard_count),
        "solved_num_bands": int(config.num_bands) + int(config.band_guard_count),
        "band_order_policy": str(config.band_order_policy),
        "eigensolver_tolerance": float(config.eigensolver_tolerance),
    }


def test_config_parse_required_provenance_fields_and_shape_derivation(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config(tmp_path))
    cfg = load_query_pilot_config(config_path)
    grid_shape = derive_grid_shape(cfg["geometry"])

    assert grid_shape == (4, 4)
    assert derive_n_wedge(grid_shape) == 3
    assert "eigensolver_tolerance" in REQUIRED_SOLVER_PROVENANCE_FIELDS
    assert "band_guard_count" in REQUIRED_SOLVER_PROVENANCE_FIELDS
    assert "band_order_policy" in REQUIRED_SOLVER_PROVENANCE_FIELDS
    assert "generation_command" in REQUIRED_SOLVER_PROVENANCE_FIELDS
    assert cfg["solver"]["band_guard_count"] == 1
    assert cfg["solver"]["band_order_policy"] == "frequency_ascending"
    assert cfg["solver"]["use_proxy"] is False


def test_no_proxy_allowed_blocks_generation(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config(tmp_path, use_proxy=True))
    run_dir = run_generation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "proxy_block_202606190000",
        skip_environment_check=True,
        solver_fn=_fake_solver,
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert results["status"] == "blocked"
    assert "solver.use_proxy must be false" in results["reason"]
    assert metrics["real_mpb_generation_started"] is False


def test_missing_tolerance_blocks_before_solver_invocation(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _counting_solver(*args, **kwargs):
        calls["count"] += 1
        return _fake_solver(*args, **kwargs)

    config_path = _write_config(tmp_path, _base_config(tmp_path, tolerance=None))
    run_dir = run_generation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "missing_tol_202606190000",
        skip_environment_check=True,
        solver_fn=_counting_solver,
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    assert calls["count"] == 0
    assert results["status"] == "blocked"
    assert "solver.eigensolver_tolerance is required" in results["reason"]
    assert "eigensolver_tolerance" in provenance["missing_required_fields"]


def test_generation_manifest_references_sidecar_and_solver_provenance(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _base_config(tmp_path))
    run_dir = run_generation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "pilot_unit_202606190000",
        skip_environment_check=True,
        solver_fn=_fake_solver,
    )

    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    manifest = yaml.safe_load((run_dir / "dataset_manifest.yaml").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    metadata_lines = (run_dir / "sample_metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert results["status"] == "pilot_generated"
    assert Path(results["dataset_path"]).name == "query_dataset_v2_pilot_unit.npz"
    assert manifest["k_path_sidecar"].endswith(".kpath.v1.json")
    assert manifest["solver_provenance"].endswith("solver_provenance.json")
    assert manifest["formal_full_2a_allowed"] is False
    assert provenance["solver_provenance_complete"] is True
    assert provenance["fields"]["band_guard_count"]["value"] == 1
    assert provenance["fields"]["band_order_policy"]["value"] == "frequency_ascending"
    assert provenance["fields"]["solved_num_bands"]["value"] == 3
    assert provenance["fields"]["eigensolver_tolerance"]["source"] == str(config_path)
    assert len(metadata_lines) == 3
    first = json.loads(metadata_lines[0])
    assert first["solver_provenance_ref"] == "solver_provenance.json"
    assert first["Wh_hash"]
    assert (run_dir / "sample_plan.jsonl").is_file()
    assert (run_dir / "worker_summary.json").is_file()


def test_generation_supports_explicit_workers_with_stable_sample_plan(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    config["solver"]["workers"] = 2
    config["solver"]["min_workers"] = 1
    config["solver"]["max_workers"] = 2
    config["solver"]["worker_backend"] = "thread"
    config_path = _write_config(tmp_path, config)

    run_dir = run_generation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "workers_unit_202606190000",
        skip_environment_check=True,
        solver_fn=_fake_solver,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "worker_summary.json").read_text(encoding="utf-8"))
    plan = [json.loads(line) for line in (run_dir / "sample_plan.jsonl").read_text(encoding="utf-8").splitlines()]
    assert metrics["workers_resolved"] == 2
    assert summary["suppress_solver_output"] is True
    assert summary["completed_count"] == 3
    assert [row["sample_id"] for row in plan] == [
        "query_dataset_v2_pilot_unit_000000",
        "query_dataset_v2_pilot_unit_000001",
        "query_dataset_v2_pilot_unit_000002",
    ]
    assert (run_dir / "worker_shards" / "query_dataset_v2_pilot_unit_000000.json").is_file()


def test_cli_help_does_not_import_meep() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "phc2d.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Unified 2D photonic-crystal workflow CLI" in proc.stdout
