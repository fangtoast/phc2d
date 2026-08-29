"""Tests for k-path physical validation scaffolding.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.data import build_square_gxm_reduced_spec, load_k_path_spec, save_k_path_spec
from phc2d.solvers import build_block_geometry_from_rho
from phc2d.solvers.mpb import _normalize_freqs
from scripts.run_k_path_physical_validation import run_validation


class _FakeMp:
    inf = float("inf")

    class Vector3:
        def __init__(self, x: float, y: float, z: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

    class Medium:
        def __init__(self, *, epsilon: float) -> None:
            self.epsilon = epsilon

    class Block:
        def __init__(self, *, center, size, material) -> None:
            self.center = center
            self.size = size
            self.material = material


def test_mpb_block_geometry_uses_inferred_grid_shape() -> None:
    rho = np.ones(36, dtype=np.float32)
    blocks, default_material, metadata = build_block_geometry_from_rho(
        _FakeMp,
        rho,
        eps_air=1.0,
        eps_material=8.9,
    )

    assert len(blocks) == 256
    assert default_material.epsilon == 1.0
    assert metadata["grid_shape"] == [16, 16]
    assert metadata["cell_size"] == 1.0 / 16.0


def test_validation_runner_blocks_before_mpb_when_tolerance_missing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.npz"
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    np.savez(
        dataset_path,
        rho=np.zeros((5, 36), dtype=np.float32),
        Wh=np.ones((5, 31, 10), dtype=np.float32),
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_val=np.array([2], dtype=np.int64),
        idx_test=np.array([3, 4], dtype=np.int64),
        config_yaml=yaml.safe_dump(
            {
                "lattice_constant": 1.0,
                "epsilon_air": 1.0,
                "epsilon_alum": 8.9,
                "geometry": {"grid_n": 16, "n_wedge": 36},
                "fem": {"n_bands": 10, "polarization": "TE"},
                "mpb": {"resolution": 64, "polarization": "TE"},
                "k_path": {"n_total": 31},
            }
        ),
    )
    sidecar = metadata_dir / "toy.kpath.v1.json"
    save_k_path_spec(build_square_gxm_reduced_spec(n_total=31), sidecar)
    config_path = tmp_path / "validate.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "k_path_physical_validation_5sample",
                "dataset": {
                    "path": str(dataset_path),
                    "manifest": str(metadata_dir / "toy.manifest.yaml"),
                    "k_path_sidecar": sidecar.name,
                },
                "sample_selection": {
                    "strategy": "deterministic_stratified",
                    "train_count": 2,
                    "val_count": 1,
                    "test_count": 2,
                    "seed": 0,
                },
                "solver": {
                    "backend": "mpb",
                    "environment": "wsl_meep_env",
                    "use_proxy": False,
                    "band_order_policy": "frequency_ascending",
                    "eigensolver_tolerance": {"value": None},
                },
                "comparison": {"compare_domain": "physical_frequency"},
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_validation(config_path, run_dir_override=tmp_path / "runs" / "20260618" / "validation_unit_202606180000")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    selected = json.loads((run_dir / "selected_samples.json").read_text(encoding="utf-8"))
    assert metrics["physical_verification_decision"] == "insufficient_provenance"
    assert metrics["real_mpb_execution_started"] is False
    assert provenance["missing_critical_fields"] == ["eigensolver_tolerance"]
    assert [row["sample_id"] for row in selected["samples"]] == ["0", "1", "2", "3", "4"]


def _v2_provenance_payload(*, tolerance: float | None = 1.0e-6) -> dict:
    fields = {
        "lattice": {"value": "square", "status": "available"},
        "reciprocal_basis": {"value": "square reciprocal basis; MPB reduced reciprocal coordinates", "status": "available"},
        "epsilon_air": {"value": 1.0, "status": "available"},
        "epsilon_material": {"value": 8.9, "status": "available"},
        "geometry_lifting_convention": {"value": "p4m_wedge_binary_pixels_as_mpb_blocks_air_background", "status": "available"},
        "grid_shape": {"value": [4, 4], "source": "geometry.grid_shape", "status": "available"},
        "n_wedge": {"value": 3, "source": "derived_from_grid_shape", "status": "available"},
        "resolution": {"value": 8, "status": "available"},
        "polarization": {"value": "TE", "status": "available"},
        "num_bands": {"value": 2, "status": "available"},
        "band_guard_count": {"value": 1, "status": "available"},
        "band_order_policy": {"value": "frequency_ascending", "status": "available"},
        "eigensolver_tolerance": {"value": tolerance, "source": "unit_v2_config.yaml", "status": "available" if tolerance is not None else "missing"},
        "frequency_normalization": {"value": "mpb_dimensionless_a_over_lambda", "status": "available"},
        "source_code_commit": {"value": "unitcommit", "status": "available"},
    }
    return {
        "schema_version": "phc2d.query_dataset_v2.solver_provenance.v1",
        "status": "pilot_smoke",
        "solver_backend": "real_mpb",
        "use_proxy": False,
        "fields": fields,
        "missing_required_fields": [] if tolerance is not None else ["eigensolver_tolerance"],
        "solver_provenance_complete": tolerance is not None,
    }


def _write_v2_fixture(tmp_path: Path, *, tolerance: float | None = 1.0e-6, use_proxy: bool = False) -> tuple[Path, Path]:
    metadata = [
        {
            "sample_id": f"query_dataset_v2_unit_{idx:06d}",
            "split": "train" if idx < 4 else "val",
            "rho_hash": f"rho{idx}",
            "geometry_hash": f"geom{idx}",
            "Wh_hash": f"wh{idx}",
        }
        for idx in range(5)
    ]
    dataset_path = tmp_path / "query_dataset_v2_unit_smoke_5.npz"
    np.savez(
        dataset_path,
        rho=np.zeros((5, 3), dtype=np.float32),
        Wh=np.zeros((5, 4, 2), dtype=np.float32),
        idx_train=np.array([0, 1, 2, 3], dtype=np.int64),
        idx_val=np.array([4], dtype=np.int64),
        idx_test=np.array([], dtype=np.int64),
        metadata_jsonl="\n".join(json.dumps(row, sort_keys=True) for row in metadata),
        config_yaml=yaml.safe_dump({"solver": {"lattice_constant": 1.0}}),
    )
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    sidecar = metadata_dir / "query_dataset_v2_unit_smoke_5.kpath.v1.json"
    save_k_path_spec(build_square_gxm_reduced_spec(n_total=4, k_path_id="query_dataset_v2_unit_square_gxm"), sidecar)
    provenance = tmp_path / "solver_provenance.json"
    provenance.write_text(json.dumps(_v2_provenance_payload(tolerance=tolerance), indent=2), encoding="utf-8")
    config_path = tmp_path / "validate_v2.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "k_path_physical_validation_v2_smoke",
                "dataset": {
                    "path": str(dataset_path),
                    "manifest": str(metadata_dir / "query_dataset_v2_unit_smoke_5.manifest.yaml"),
                    "k_path_sidecar": sidecar.name,
                },
                "sample_selection": {
                    "strategy": "all_available_smoke_samples",
                    "train_count": 4,
                    "val_count": 1,
                    "test_count": 0,
                    "rule": "all_available_v2_smoke_samples_train4_val1_test0",
                },
                "solver": {
                    "backend": "mpb",
                    "environment": "wsl_meep_env",
                    "use_proxy": use_proxy,
                    "provenance_path": str(provenance),
                },
                "comparison": {"compare_domain": "physical_frequency", "repeat_first_sample_count": 2},
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path, dataset_path


def test_v2_smoke_validation_parses_solver_provenance_and_zero_test_split(tmp_path: Path) -> None:
    calls = {"grid_shape": None, "num_bands": None, "band_guard_count": None, "band_order_policy": None, "nk": None, "count": 0}

    def _fake_solver(rho, *, k_points, config):
        calls["grid_shape"] = config.grid_shape
        calls["num_bands"] = config.num_bands
        calls["band_guard_count"] = config.band_guard_count
        calls["band_order_policy"] = config.band_order_policy
        calls["nk"] = len(k_points)
        calls["count"] += 1
        return np.zeros((len(k_points), int(config.num_bands)), dtype=np.float32), {
            "mpb_backend": "fake_unit_solver",
            "grid_shape": list(config.grid_shape),
            "num_bands": int(config.num_bands),
            "band_guard_count": int(config.band_guard_count),
            "band_order_policy": str(config.band_order_policy),
            "solved_num_bands": int(config.num_bands) + int(config.band_guard_count),
            "eigensolver_tolerance": float(config.eigensolver_tolerance),
        }

    config_path, _ = _write_v2_fixture(tmp_path)
    run_dir = run_validation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "validation_v2_unit_202606190000",
        solver_fn=_fake_solver,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    selected = json.loads((run_dir / "selected_samples.json").read_text(encoding="utf-8"))
    assert metrics["physical_verification_decision"] == "physically_verified_with_documented_numerical_tolerance"
    assert metrics["real_mpb_execution_completed"] is True
    assert metrics["sample_count"] == 5
    assert provenance["solver_backend"] == "real_mpb"
    assert provenance["fields"]["grid_shape"]["value"] == [4, 4]
    assert provenance["fields"]["n_wedge"]["value"] == 3
    assert provenance["fields"]["band_guard_count"]["value"] == 1
    assert provenance["fields"]["band_order_policy"]["value"] == "frequency_ascending"
    assert calls == {"grid_shape": (4, 4), "num_bands": 2, "band_guard_count": 1, "band_order_policy": "frequency_ascending", "nk": 4, "count": 6}
    assert [row["split"] for row in selected["samples"]] == ["train", "train", "train", "train", "val"]
    assert selected["samples"][-1]["sample_id"] == "query_dataset_v2_unit_000004"
    assert (run_dir / "worker_summary.json").is_file()


def test_v2_validation_supports_explicit_workers(tmp_path: Path) -> None:
    def _fake_solver(rho, *, k_points, config):
        return np.zeros((len(k_points), int(config.num_bands)), dtype=np.float32), {
            "mpb_backend": "fake_unit_solver",
            "eigensolver_tolerance": float(config.eigensolver_tolerance),
        }

    config_path, _ = _write_v2_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["solver"]["workers"] = 2
    payload["solver"]["min_workers"] = 1
    payload["solver"]["max_workers"] = 2
    payload["solver"]["worker_backend"] = "thread"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    run_dir = run_validation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "validation_workers_unit_202606190000",
        solver_fn=_fake_solver,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["workers_resolved"] == 2
    assert metrics["completed_count"] == 5
    assert metrics["physical_verification_decision"] == "physically_verified_with_documented_numerical_tolerance"


def test_v2_validation_promotes_sidecar_and_manifest_when_requested(tmp_path: Path) -> None:
    def _fake_solver(rho, *, k_points, config):
        return np.zeros((len(k_points), int(config.num_bands)), dtype=np.float32), {
            "mpb_backend": "fake_unit_solver",
            "eigensolver_tolerance": float(config.eigensolver_tolerance),
        }

    config_path, dataset_path = _write_v2_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = tmp_path / "metadata" / "query_dataset_v2_unit_smoke_5.manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "dataset_path": str(dataset_path),
                "k_path_sidecar": "query_dataset_v2_unit_smoke_5.kpath.v1.json",
                "solver_provenance": str(tmp_path / "solver_provenance.json"),
                "formal_full_2a_allowed": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload["dataset"]["manifest"] = str(manifest)
    payload["solver"].pop("provenance_path")
    payload["output"]["promote_verified_sidecar"] = True
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    run_dir = run_validation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "promote_v2_unit_202606190000",
        solver_fn=_fake_solver,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    promoted = load_k_path_spec(tmp_path / "metadata" / "query_dataset_v2_unit_smoke_5.kpath.v1.json")
    manifest_payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert metrics["sidecar_manifest_status"] == "promoted"
    assert promoted.metadata.verification_status == "physically_verified_with_documented_numerical_tolerance"
    assert promoted.metadata.verification_run_id == run_dir.name
    assert manifest_payload["formal_full_2a_allowed"] is True
    assert manifest_payload["physical_verification_status"] == "physically_verified_with_documented_numerical_tolerance"


def test_v2_validation_proxy_replay_is_forbidden(tmp_path: Path) -> None:
    config_path, _ = _write_v2_fixture(tmp_path, use_proxy=True)
    with pytest.raises(ValueError, match="proxy replay is forbidden"):
        run_validation(config_path, run_dir_override=tmp_path / "runs" / "20260619" / "proxy_block_202606190000")


def test_v2_validation_blocks_before_mpb_when_frozen_dataset_identity_mismatches(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _fake_solver(*args, **kwargs):
        calls["count"] += 1
        return np.zeros((4, 2), dtype=np.float32), {}

    config_path, _ = _write_v2_fixture(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["dataset"]["expected_dataset_sha256"] = "0" * 64
    payload["dataset"]["expected_split_sizes"] = {"train": 4, "val": 1, "test": 0}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    run_dir = run_validation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "identity_mismatch_v2_202606190000",
        solver_fn=_fake_solver,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert calls["count"] == 0
    assert metrics["physical_verification_decision"] == "dataset_identity_mismatch"
    assert "dataset_sha256_mismatch" in results["reason"]


def test_mpb_frequency_normalization_canonicalizes_band_order() -> None:
    freqs = np.asarray(
        [
            [0.1, 0.3, 0.2],
            [0.4, 0.35, 0.5],
        ],
        dtype=np.float64,
    )

    normalized = _normalize_freqs(freqs, n_k=2, n_bands=3, band_order_policy="frequency_ascending")

    assert normalized.tolist() == [[0.1, 0.2, 0.3], [0.35, 0.4, 0.5]]


def test_mpb_frequency_normalization_keeps_square_k_by_band_orientation() -> None:
    frequencies = np.asarray(
        [
            [0.30, 0.10, 0.20],
            [0.60, 0.40, 0.50],
            [0.90, 0.70, 0.80],
        ],
        dtype=np.float64,
    )

    normalized = _normalize_freqs(
        frequencies,
        n_k=3,
        n_bands=3,
        band_order_policy="frequency_ascending",
    )

    assert normalized.tolist() == [
        [0.10, 0.20, 0.30],
        [0.40, 0.50, 0.60],
        [0.70, 0.80, 0.90],
    ]


def test_v2_validation_blocks_when_tolerance_missing_before_solver(tmp_path: Path) -> None:
    calls = {"count": 0}

    def _fake_solver(*args, **kwargs):
        calls["count"] += 1
        return np.zeros((4, 2), dtype=np.float32), {}

    config_path, _ = _write_v2_fixture(tmp_path, tolerance=None)
    run_dir = run_validation(
        config_path,
        run_dir_override=tmp_path / "runs" / "20260619" / "missing_tol_v2_202606190000",
        solver_fn=_fake_solver,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "solver_provenance.json").read_text(encoding="utf-8"))
    assert calls["count"] == 0
    assert metrics["physical_verification_decision"] == "insufficient_provenance"
    assert metrics["real_mpb_execution_started"] is False
    assert provenance["missing_critical_fields"] == ["eigensolver_tolerance"]
