"""Tests for Query Dataset V2 smoke mismatch artifact audit.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from phc2d.data.forward_dataset import serialize_metadata_jsonl
from phc2d.data.k_path import build_square_gxm_reduced_spec, save_k_path_spec
from phc2d.data.lift import lift_rho_to_grid
from scripts.run_v2_smoke_guard_band_diagnostic import run_diagnostic
from scripts.run_v2_smoke_mismatch_audit import _generation_hash_array, run_audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _provenance_pair() -> tuple[dict, dict]:
    common_fields = {
        "epsilon_air": {"value": 1.0, "status": "available"},
        "epsilon_material": {"value": 8.9, "status": "available"},
        "geometry_lifting_convention": {"value": "p4m_wedge_binary_pixels_as_mpb_blocks_air_background", "status": "available"},
        "grid_shape": {"value": [4, 4], "status": "available"},
        "n_wedge": {"value": 3, "status": "available"},
        "resolution": {"value": 8, "status": "available"},
        "polarization": {"value": "TE", "status": "available"},
        "num_bands": {"value": 3, "status": "available"},
        "eigensolver_tolerance": {"value": 1.0e-7, "status": "available"},
        "frequency_normalization": {"value": "mpb_dimensionless_a_over_lambda", "status": "available"},
    }
    generation = {
        "solver_backend": "real_mpb",
        "use_proxy": False,
        "fields": {
            **common_fields,
            "k_points": {"value": {"count": 4, "hash": "unit-k-points-hash"}, "status": "available"},
        },
        "missing_required_fields": [],
    }
    validation = {
        "solver_backend": "real_mpb",
        "use_proxy": False,
        "fields": {
            **common_fields,
            "k_path": {"value": {"Nk": 4, "k_points_hash": "unit-k-points-hash"}, "status": "available"},
        },
        "missing_critical_fields": [],
    }
    return generation, validation


def _fixture(tmp_path: Path, *, corrupt_npz_metadata: bool = False) -> tuple[Path, Path, Path]:
    generation_run = tmp_path / "runs" / "20260619" / "query_dataset_v2_pilot_unit_202606190000"
    validation_run = tmp_path / "runs" / "20260619" / "k_path_physical_validation_unit_202606190001"
    generation_run.mkdir(parents=True)
    (validation_run / "stored_bands").mkdir(parents=True)
    (validation_run / "recomputed_bands").mkdir(parents=True)

    rho = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    wh = np.zeros((2, 4, 3), dtype=np.float32)
    metadata = []
    selected_rows = []
    for idx in range(2):
        grid = (lift_rho_to_grid(rho[idx], grid_shape=(4, 4)) > 0.5).astype(np.uint8)
        row = {
            "sample_id": f"query_dataset_v2_unit_{idx:06d}",
            "split": "train",
            "sample_index": idx,
            "rho_hash": _generation_hash_array(rho[idx]),
            "geometry_hash": _generation_hash_array(grid),
            "Wh_hash": _generation_hash_array(wh[idx]),
        }
        metadata.append(dict(row))
        selected_rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset_index": idx,
                "split": "train",
                "split_position": idx,
                "rho_hash": row["rho_hash"],
                "geometry_hash": row["geometry_hash"],
                "existing_Wh_hash": row["Wh_hash"],
                "fill_fraction": 0.5,
            }
        )

    npz_metadata = [dict(row) for row in metadata]
    if corrupt_npz_metadata:
        npz_metadata[0]["Wh_hash"] = "stale_npz_wh_hash"
    dataset_path = generation_run / "query_dataset_v2_unit_smoke_2.npz"
    np.savez_compressed(
        dataset_path,
        rho=rho,
        Wh=wh,
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_val=np.array([], dtype=np.int64),
        idx_test=np.array([], dtype=np.int64),
        metadata_jsonl=serialize_metadata_jsonl(npz_metadata),
        config_yaml="unit: true\n",
    )
    (generation_run / "sample_metadata.jsonl").write_text(serialize_metadata_jsonl(metadata) + "\n", encoding="utf-8")
    _write_csv(
        validation_run / "selected_samples.csv",
        selected_rows,
        [
            "sample_id",
            "dataset_index",
            "split",
            "split_position",
            "rho_hash",
            "geometry_hash",
            "existing_Wh_hash",
            "fill_fraction",
        ],
    )
    stored0 = wh[0].copy()
    stored1 = wh[1].copy()
    recomputed0 = stored0.copy()
    recomputed1 = stored1.copy()
    recomputed1[1, 2] = 0.015
    recomputed1[2, 2] = 0.031
    np.save(validation_run / "stored_bands" / "sample_0_stored.npy", stored0)
    np.save(validation_run / "recomputed_bands" / "sample_0_recomputed.npy", recomputed0)
    np.save(validation_run / "stored_bands" / "sample_1_stored.npy", stored1)
    np.save(validation_run / "recomputed_bands" / "sample_1_recomputed.npy", recomputed1)

    generation_provenance, validation_provenance = _provenance_pair()
    _write_json(generation_run / "solver_provenance.json", generation_provenance)
    _write_json(validation_run / "solver_provenance.json", validation_provenance)
    _write_json(generation_run / "results.json", {"dataset_path": str(dataset_path)})
    return generation_run, validation_run, dataset_path


def test_audit_identifies_localized_high_band_mismatch(tmp_path: Path) -> None:
    generation_run, validation_run, dataset_path = _fixture(tmp_path)

    run_dir = run_audit(
        validation_run=validation_run,
        generation_run=generation_run,
        dataset_path=dataset_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "audit_unit_202606190002",
    )

    summary = json.loads((run_dir / "mismatch_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert summary["status"] == "mismatch_root_cause_identified"
    assert summary["root_cause_family"] == "localized_generation_stored_Wh_high_band_replay_divergence"
    assert summary["identity_all_match"] is True
    assert summary["provenance_matches"] is True
    assert summary["mismatch"]["significant_samples"] == ["query_dataset_v2_unit_000001"]
    assert summary["mismatch"]["significant_bands"] == [2]
    assert summary["mismatch"]["significant_k_indices"] == [1, 2]
    assert metrics["max_abs_error"] == pytest.approx(0.031)
    assert metrics["formal_full_2a_allowed"] is False


def test_audit_marks_zero_significant_mismatch_as_fixed(tmp_path: Path) -> None:
    generation_run, validation_run, dataset_path = _fixture(tmp_path)
    stored = np.load(validation_run / "stored_bands" / "sample_1_stored.npy")
    np.save(validation_run / "recomputed_bands" / "sample_1_recomputed.npy", stored)

    run_dir = run_audit(
        validation_run=validation_run,
        generation_run=generation_run,
        dataset_path=dataset_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "audit_fixed_unit_202606190004",
    )

    summary = json.loads((run_dir / "mismatch_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert summary["status"] == "fixed_and_v2_smoke_validation_passed"
    assert summary["root_cause_family"] == "no_significant_direct_alignment_mismatch"
    assert summary["formal_full_2a_allowed"] is True
    assert metrics["significant_cell_count"] == 0
    assert metrics["formal_full_2a_allowed"] is True


def test_audit_recomputes_npz_identity_hashes(tmp_path: Path) -> None:
    generation_run, validation_run, dataset_path = _fixture(tmp_path, corrupt_npz_metadata=True)

    run_dir = run_audit(
        validation_run=validation_run,
        generation_run=generation_run,
        dataset_path=dataset_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "audit_identity_unit_202606190003",
    )

    summary = json.loads((run_dir / "mismatch_summary.json").read_text(encoding="utf-8"))
    identity_rows = list(csv.DictReader((run_dir / "sample_identity_audit.csv").open(newline="", encoding="utf-8")))
    assert summary["status"] == "mismatch_root_cause_identified"
    assert summary["root_cause_family"] == "sample_identity_or_dataset_association_mismatch"
    assert summary["identity_all_match"] is False
    assert identity_rows[0]["Wh_hash_match"] == "False"
    assert identity_rows[0]["computed_Wh_hash"] != "stale_npz_wh_hash"


def test_guard_band_diagnostic_records_guard_effect(tmp_path: Path) -> None:
    generation_run = tmp_path / "runs" / "20260619" / "query_dataset_v2_guard_unit_202606190000"
    metadata_dir = generation_run / "metadata"
    metadata_dir.mkdir(parents=True)
    dataset_path = generation_run / "query_dataset_v2_guard_unit_smoke_1.npz"
    sidecar_path = metadata_dir / "query_dataset_v2_guard_unit_smoke_1.kpath.v1.json"
    rho = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    wh = np.ones((1, 4, 2), dtype=np.float32)
    np.savez_compressed(
        dataset_path,
        rho=rho,
        Wh=wh,
        idx_train=np.array([0], dtype=np.int64),
        idx_val=np.array([], dtype=np.int64),
        idx_test=np.array([], dtype=np.int64),
        metadata_jsonl=serialize_metadata_jsonl([{"sample_id": "sample_000000"}]),
        config_yaml="unit: true\n",
    )
    save_k_path_spec(build_square_gxm_reduced_spec(n_total=4, k_path_id="query_dataset_v2_guard_unit_square_gxm"), sidecar_path)
    _write_json(
        generation_run / "solver_provenance.json",
        {
            "solver_backend": "real_mpb",
            "use_proxy": False,
            "fields": {
                "epsilon_air": {"value": 1.0},
                "epsilon_material": {"value": 8.9},
                "grid_shape": {"value": [4, 4]},
                "resolution": {"value": 8},
                "polarization": {"value": "TE"},
                "num_bands": {"value": 2},
                "eigensolver_tolerance": {"value": 1.0e-7},
            },
        },
    )
    _write_json(generation_run / "results.json", {"dataset_path": str(dataset_path)})

    def _fake_solver(rho, *, k_points, config):
        value = 1.0 if int(config.band_guard_count) == 0 else 0.75
        return np.full((len(k_points), int(config.num_bands)), value, dtype=np.float32), {
            "num_bands": int(config.num_bands),
            "band_guard_count": int(config.band_guard_count),
            "solved_num_bands": int(config.num_bands) + int(config.band_guard_count),
        }

    run_dir = run_diagnostic(
        generation_run=generation_run,
        sample_index=0,
        k_indices=[0, 1],
        guard_counts=[0, 2],
        run_dir_override=tmp_path / "runs" / "20260619" / "guard_diag_unit_202606190001",
        solver_fn=_fake_solver,
    )

    summary = json.loads((run_dir / "guard_band_summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_dir / "guard_band_runs.csv").open(newline="", encoding="utf-8")))
    assert summary["status"] == "guard_band_diagnostic_completed"
    assert summary["root_cause_candidate"] == "top_retained_band_guard_missing"
    assert summary["max_abs_by_guard"]["0"] == 0.0
    assert summary["max_abs_by_guard"]["2"] == 0.25
    assert rows[1]["solved_num_bands"] == "4"
