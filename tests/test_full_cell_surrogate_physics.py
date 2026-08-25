"""Tests for Full-Wh spectral and real-response sensitivity evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from phc2d.data import build_full_bz_grid
from phc2d.generators import generate_binary_polar_holes
from phc2d.plot.full_cell_surrogate_physics import (
    build_full_cell_surrogate_physics,
    paired_bootstrap_mean_difference,
    response_difference_metrics,
    select_quantile_positions,
)
from phc2d.solvers import MpbEigenfieldResult
from phc2d.solvers.meep_scattering import MeepScatteringConfig, MeepScatteringResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generator_config() -> dict[str, object]:
    return {
        "epsilon_air": 1.0,
        "epsilon_material": 8.9,
        "audit_resolution": 16,
        "fourier_cutoff": 3,
        "spectral_cutoff_scale_range": [2.5, 4.5],
        "graded_target_material_fraction_range": [0.60, 0.78],
        "graded_temperature_range": [0.08, 0.16],
        "graded_fourier_strength_range": [0.03, 0.10],
        "graded_air_well_magnitude_range": [0.90, 1.40],
        "max_attempts": 512,
    }


def _write_hdf5(path: Path, *, lane: str, count: int) -> tuple[np.ndarray, list[object]]:
    base = generate_binary_polar_holes(
        f"{lane.lower()}_0",
        master_seed=17,
        hole_count=1,
        config=_generator_config(),
    )
    structures = [replace(base, sample_id=f"{lane.lower()}_{index}") for index in range(count)]
    k_points, k_grid_index = build_full_bz_grid(9)
    target = np.empty((count, 81, 10), dtype=np.float32)
    for sample in range(count):
        target[sample] = np.arange(1, 11, dtype=np.float32)[None, :] * 0.1 + sample * 0.001
    utf8 = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "phc2d.full_cell_dataset.v2"
        handle.create_dataset("idx_test", data=np.arange(count, dtype=np.int64))
        handle.create_dataset("Wh", data=target)
        handle.create_dataset("k_points", data=k_points)
        handle.create_dataset("k_grid_index", data=k_grid_index)
        handle.create_dataset("sample_id", data=np.asarray([item.sample_id for item in structures], dtype=object), dtype=utf8)
        handle.create_dataset("geometry_hash", data=np.asarray([item.geometry_hash for item in structures], dtype=object), dtype=utf8)
        handle.create_dataset(
            "structure_json",
            data=np.asarray([json.dumps(item.to_mapping(), sort_keys=True) for item in structures], dtype=object),
            dtype=utf8,
        )
        grid = np.stack([item.rasterize((8, 8)) for item in structures]).astype(np.float32)
        handle.create_dataset("material_area_fraction_grid_64", data=grid)
        handle.create_dataset("material_weight_grid_64", data=grid)
    return target, structures


def _write_inputs(tmp_path: Path) -> Path:
    count = 8
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    datasets: dict[str, dict[str, str]] = {}
    evaluation_rows: list[dict[str, object]] = []
    for lane in ("B", "G"):
        dataset_path = tmp_path / f"{lane}.h5"
        target, _ = _write_hdf5(dataset_path, lane=lane, count=count)
        datasets[lane] = {
            "target_domain": f"K5-{lane}",
            "display_name": f"K5 {lane}",
            "path": str(dataset_path),
            "grid_field": "material_area_fraction_grid_64" if lane == "B" else "material_weight_grid_64",
            "expected_dataset_sha256": _sha256(dataset_path),
        }
        for model_index, (model_key, source_model, source_lane) in enumerate(
            (("convnext", "convnext_band_v1", lane), ("joint", "dual_lane_band_operator_v1", "J"))
        ):
            shard_id = 2 * (0 if lane == "B" else 1) + model_index
            evaluation_id = f"eval_{lane}_{model_key}"
            shard = campaign / "shards" / f"shard_{shard_id:02d}"
            shard.mkdir(parents=True)
            scale = (np.arange(count, dtype=np.float32) + 1.0) * (0.001 if model_key == "convnext" else 0.0008)
            prediction = target * (1.0 + scale[:, None, None])
            sample_mre = 100.0 * scale
            np.savez_compressed(shard / f"{evaluation_id}.npz", prediction=prediction, sample_mre=sample_mre, target=target)
            evaluation_rows.append(
                {
                    "source_family": "K5",
                    "source_seed": 0,
                    "source_model": source_model,
                    "source_lane": source_lane,
                    "target_domain": f"K5-{lane}",
                    "status": "completed",
                    "shard_id": shard_id,
                    "evaluation_id": evaluation_id,
                    "checkpoint_sha256": f"checkpoint-{lane}-{model_key}",
                }
            )
    with (campaign / "evaluation_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluation_rows[0]))
        writer.writeheader()
        writer.writerows(evaluation_rows)
    config = {
        "experiment": {"experiment_id": "EXP-PHY-004", "family": "research", "stage": "test", "primary_layer": "L3"},
        "evidence": {
            "campaign_dir": str(campaign),
            "source_family": "K5",
            "source_seed": 0,
            "models": {
                "convnext": {"source_model": "convnext_band_v1", "source_lane": "match_target"},
                "joint": {"source_model": "dual_lane_band_operator_v1", "source_lane": "J"},
            },
        },
        "datasets": datasets,
        "sampling": {"split": "test", "expected_test_count": count, "quantiles": [0.1, 0.5, 0.9]},
        "bands": {"gap_lower_bands_one_based": [2, 9]},
        "statistics": {"paired_bootstrap_replicates": 100, "bootstrap_seed": 5},
        "mpb_eigenfield": {
            "epsilon_air": 1.0,
            "epsilon_material": 8.9,
            "resolution": 8,
            "mesh_size": 1,
            "polarization": "TE",
            "num_bands": 10,
            "band_guard_count": 0,
            "band_order_policy": "frequency_ascending",
            "eigensolver_tolerance": 1.0e-7,
            "frequency_mre_max": 0.01,
        },
        "meep_response": {
            "periods_x": 3,
            "periods_y": 3,
            "resolution": 8,
            "pml_thickness": 1.0,
            "monitor_padding": 0.5,
            "source_gap": 0.5,
            "fractional_bandwidth": 0.1,
            "dft_decay_tolerance": 1.0e-5,
            "dft_minimum_run_time": 1.0,
            "dft_maximum_run_time": 2.0,
            "require_decay_before_max": True,
            "farfield_radius_wavelengths": 10.0,
            "farfield_angle_count": 24,
            "greencyl_tolerance": 1.0e-4,
        },
        "runtime": {"authorized_solver_execution": True},
        "output": {"runs_dir": str(tmp_path / "runs"), "formats": ["svg"], "dpi": 72},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _fake_eigenfield_solver(structure, *, k_point, band_index_one_based, config) -> MpbEigenfieldResult:
    field = np.ones((8, 8, 1, 3), dtype=np.complex128)
    return MpbEigenfieldResult(
        eigenfrequency=float(band_index_one_based) * 0.1 + float(structure.sample_id.split("_")[-1]) * 0.001,
        efield=field,
        hfield=field * (0.2 + 0.1j),
        requested_band_index_one_based=int(band_index_one_based),
        solver_band_index_one_based=int(band_index_one_based),
        k_point=tuple(float(value) for value in k_point),
        metadata={"solver_backend": "deterministic_fake", "sample_id": structure.sample_id},
    )


def _fake_scattering_solver(structure, *, frequency, config: MeepScatteringConfig) -> MeepScatteringResult:
    x = np.linspace(-2.0, 2.0, 12)
    y = np.linspace(-2.0, 2.0, 12)
    xx, yy = np.meshgrid(x, y)
    energy = np.exp(-((xx - 0.25 * float(frequency)) ** 2 + yy**2))
    epsilon = np.where(xx < 0.0, 1.0, 8.9)
    angles = np.linspace(-np.pi, np.pi, config.farfield_angle_count, endpoint=False)
    far = 0.2 + np.cos(angles - float(frequency)) ** 2
    return MeepScatteringResult(
        float(frequency),
        x,
        y,
        epsilon,
        energy,
        np.ones_like(energy),
        0.2 * np.sin(yy),
        angles,
        far,
        0.01,
        {"geometry_hash": structure.geometry_hash, "response_config": asdict(config)},
    )


def test_quantile_bootstrap_and_identical_response_metrics() -> None:
    selected = select_quantile_positions(np.arange(10.0), [0.1, 0.5, 0.9])
    assert [item[1] for item in selected] == [1, 4, 8]
    interval = paired_bootstrap_mean_difference(np.ones(6), np.zeros(6), replicates=100, seed=1)
    assert interval["mean_difference_pct_points"] == pytest.approx(1.0)
    x = y = np.linspace(-1.0, 1.0, 4)
    angles = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    response = {
        "frequency": 0.4,
        "x": x,
        "y": y,
        "electric_energy": np.ones((4, 4)),
        "far_angles_rad": angles,
        "far_radial_flux": np.ones(12),
    }
    metrics = response_difference_metrics(response, response)
    assert metrics["near_relative_l2"] == pytest.approx(0.0)
    assert metrics["far_angular_l1"] == pytest.approx(0.0)
    assert metrics["near_cosine_similarity"] == pytest.approx(1.0)


def test_end_to_end_spectral_preflight_and_full_physics(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    config = _write_inputs(tmp_path)
    run_dir = tmp_path / "run"
    build_full_cell_surrogate_physics(config, run_dir=run_dir, stage="spectral")
    assert len(list((run_dir / "figures").glob("*.svg"))) == 5
    assert len(list(csv.DictReader((run_dir / "sample_metrics.csv").open(encoding="utf-8")))) == 16
    assert len(list(csv.DictReader((run_dir / "selected_samples.csv").open(encoding="utf-8")))) == 6

    build_full_cell_surrogate_physics(
        config,
        run_dir=run_dir,
        stage="physics",
        physics_quantiles=[0.5],
        eigenfield_solver=_fake_eigenfield_solver,
        scattering_solver=_fake_scattering_solver,
        skip_environment_check=True,
    )
    preflight = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "p50_preflight_complete"
    assert preflight["physical_sample_count_completed"] == 2

    build_full_cell_surrogate_physics(
        config,
        run_dir=run_dir,
        stage="physics",
        eigenfield_solver=_fake_eigenfield_solver,
        scattering_solver=_fake_scattering_solver,
        skip_environment_check=True,
    )
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert results["status"] == "completed"
    assert results["physical_solver_execution_completed"] is True
    assert metrics["mpb_eigenfield_logical_case_count"] == 6
    assert metrics["meep_response_logical_case_count"] == 18
    assert len(list((run_dir / "mpb_field_shards").glob("*.npz"))) == 6
    assert len(list((run_dir / "response_shards").glob("*.npz"))) == 18
    assert len(list(csv.DictReader((run_dir / "response_difference_metrics.csv").open(encoding="utf-8")))) == 12


def test_surrogate_physics_cli_help_does_not_import_meep(monkeypatch: pytest.MonkeyPatch) -> None:
    from phc2d.cli import main

    monkeypatch.setitem(__import__("sys").modules, "meep", None)
    with pytest.raises(SystemExit) as exc:
        main(["plot", "full-cell-surrogate-physics", "--help"])
    assert exc.value.code == 0
