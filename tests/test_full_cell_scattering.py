"""Tests for finite-array Meep near/far-field response galleries.

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
import yaml

from phc2d.data import build_full_bz_grid, write_full_cell_hdf5
from phc2d.generators import generate_binary_polar_holes, generate_graded_rbf_fourier
from phc2d.plot.full_cell_scattering import (
    build_full_cell_scattering_gallery,
    normalize_response_for_display,
    poynting_quiver_samples,
)
from phc2d.solvers.meep_scattering import (
    MeepScatteringConfig,
    MeepScatteringResult,
    _radial_poynting_from_farfield,
)


def test_radial_poynting_and_display_normalization() -> None:
    angles = np.asarray([0.0, 0.5 * np.pi])
    fields = np.zeros((2, 6), dtype=np.complex128)
    fields[0, 1], fields[0, 5] = 1.0, 2.0
    fields[1, 0], fields[1, 5] = -1.0, 2.0
    assert np.allclose(_radial_poynting_from_farfield(fields, angles), [1.0, 1.0])
    near, far = normalize_response_for_display(np.asarray([[0.0, 2.0]]), np.asarray([1.0, 4.0]))
    assert np.allclose(near, [[0.0, 1.0]])
    assert np.allclose(far, [0.25, 1.0])


def test_poynting_quiver_samples_preserve_direction_and_mask_weak_vectors() -> None:
    x = np.linspace(-1.0, 1.0, 8)
    y = np.linspace(-1.0, 1.0, 6)
    sx = np.ones((6, 8))
    sy = np.zeros((6, 8))
    sx[0, 0] = 0.01
    qx, qy, u, v = poynting_quiver_samples(x, y, sx, sy, target_count=4, relative_threshold=0.1)
    assert qx.shape == qy.shape == u.shape == v.shape
    assert np.isnan(u[0, 0])
    assert np.nanmin(u) == pytest.approx(1.0)
    assert np.nanmax(np.abs(v)) == pytest.approx(0.0)


def _generator_config() -> dict[str, object]:
    return {
        "epsilon_air": 1.0,
        "epsilon_material": 8.9,
        "audit_resolution": 24,
        "fourier_cutoff": 3,
        "spectral_cutoff_scale_range": [2.5, 4.5],
        "graded_target_material_fraction_range": [0.60, 0.78],
        "graded_temperature_range": [0.08, 0.16],
        "graded_fourier_strength_range": [0.03, 0.10],
        "graded_air_well_magnitude_range": [0.90, 1.40],
        "max_attempts": 512,
    }


def _write_inputs(tmp_path: Path) -> Path:
    source_run = tmp_path / "source_gallery"
    source_run.mkdir()
    structures = {
        "graded_medium": generate_graded_rbf_fourier("graded_0", master_seed=7, core_count=1, config=_generator_config()),
        "binary_air_holes": generate_binary_polar_holes("binary_0", master_seed=7, hole_count=1, config=_generator_config()),
    }
    k_points, k_index = build_full_bz_grid(9)
    bands = np.tile(np.linspace(0.1, 1.0, 10), (1, 81, 1)).astype(np.float32)
    datasets = []
    with (source_run / "selected_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lane", "sample_label", "sample_id", "dataset_index"])
        writer.writeheader()
        for lane, structure in structures.items():
            path = tmp_path / f"{lane}.h5"
            write_full_cell_hdf5(path, structures=[structure], Wh=bands, k_points=k_points, k_grid_index=k_index, idx_train=np.asarray([0]), idx_val=np.asarray([], dtype=int), idx_test=np.asarray([], dtype=int), raster_resolutions=[16], metadata={})
            datasets.append({"lane": lane, "path": str(path), "status": "test_only"})
            writer.writerow({"lane": lane, "sample_label": "S01", "sample_id": structure.sample_id, "dataset_index": 0})
    config = {
        "experiment": {"experiment_id": "meep_scattering_test", "family": "research", "stage": "meep_scattering_response", "primary_layer": "L3"},
        "source_gallery_run": str(source_run),
        "datasets": datasets,
        "bands": {"gap_lower_bands_one_based": [2, 9]},
        "response": {"periods_x": 3, "periods_y": 3, "resolution": 8, "pml_thickness": 1.0, "monitor_padding": 0.5, "source_gap": 0.5, "fractional_bandwidth": 0.1, "dft_decay_tolerance": 1e-5, "dft_minimum_run_time": 1.0, "dft_maximum_run_time": 2.0, "require_decay_before_max": True, "farfield_radius_wavelengths": 10.0, "farfield_angle_count": 24, "greencyl_tolerance": 1e-4},
        "output": {"runs_dir": str(tmp_path / "runs"), "formats": ["png", "svg", "pdf"], "dpi": 72, "arrow_count": 4, "arrow_relative_threshold": 0.05},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _fake_solver(structure, *, frequency, config: MeepScatteringConfig) -> MeepScatteringResult:
    x = np.linspace(-2.0, 2.0, 12)
    y = np.linspace(-2.0, 2.0, 12)
    xx, yy = np.meshgrid(x, y)
    energy = np.exp(-(xx * xx + yy * yy))
    epsilon = np.where(xx < 0.0, 1.0, 8.9)
    angles = np.linspace(-np.pi, np.pi, config.farfield_angle_count, endpoint=False)
    return MeepScatteringResult(float(frequency), x, y, epsilon, energy, np.ones_like(energy), 0.2 * np.sin(yy), angles, 0.2 + np.cos(angles) ** 2, 0.01, {"farfield_negative_flux_fraction_before_clipping": 0.0, "geometry_hash": structure.geometry_hash, "response_config": dict(config.__dict__)})


def test_scattering_gallery_runner_writes_near_far_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    config = _write_inputs(tmp_path)
    out = build_full_cell_scattering_gallery(config, run_dir=tmp_path / "run", solver_fn=_fake_solver, skip_environment_check=True)
    for name in ("run.log", "run.meta.json", "config.resolved.yaml", "results.json", "metrics.json", "history.csv", "response_records.csv", "scattering_response_records.h5", "figure_manifest.json"):
        assert (out / name).is_file(), name
    assert len(list((out / "figures").glob("*"))) == 15
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["sample_count_total"] == 2
    assert results["response_shard_count"] == 2
    assert "Poynting" in results["near_field_quantity"]
    assert results["source_gallery_acceptance_advanced"] is False

    def fail_if_called(*args, **kwargs):
        raise AssertionError("valid response shards should be reused")

    build_full_cell_scattering_gallery(config, run_dir=tmp_path / "run", solver_fn=fail_if_called, skip_environment_check=True)
    rerun_metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert rerun_metrics["meep_executed_this_invocation_count"] == 0
    assert rerun_metrics["response_shard_reuse_count"] == 2


def test_scattering_cli_help_does_not_import_meep(monkeypatch: pytest.MonkeyPatch) -> None:
    from phc2d.cli import main

    monkeypatch.setitem(__import__("sys").modules, "meep", None)
    with pytest.raises(SystemExit) as exc:
        main(["plot", "full-cell-scattering-gallery", "--help"])
    assert exc.value.code == 0
