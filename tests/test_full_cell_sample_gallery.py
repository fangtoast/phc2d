"""Tests for traceable full-cell dielectric, band, and eigenfield galleries.

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

import phc2d.solvers.full_cell as full_cell_solver
from phc2d.data import build_full_bz_grid, write_full_cell_hdf5
from phc2d.generators import generate_binary_polar_holes, generate_binary_spectral_threshold, generate_graded_rbf_fourier
from phc2d.plot.full_cell_gallery import (
    build_full_cell_sample_gallery,
    infer_full_cell_stratum,
    normalized_electric_energy_density,
    sampled_gxm_path,
    select_largest_gap_edge,
    select_stratified_test_indices,
)
from phc2d.solvers import FullCellMpbBandSolverConfig, MpbEigenfieldResult, solve_eigenfield_for_full_cell_mpb


def _generator_config() -> dict[str, object]:
    return {
        "epsilon_air": 1.0,
        "epsilon_material": 8.9,
        "audit_resolution": 32,
        "fourier_cutoff": 3,
        "spectral_cutoff_scale_range": [2.5, 4.5],
        "graded_target_material_fraction_range": [0.60, 0.78],
        "graded_temperature_range": [0.08, 0.16],
        "graded_fourier_strength_range": [0.03, 0.10],
        "graded_air_well_magnitude_range": [0.90, 1.40],
        "max_attempts": 512,
    }


def _structures() -> tuple[list, list]:
    graded = [
        generate_graded_rbf_fourier(
            f"graded_gallery_{index:06d}",
            master_seed=20260806,
            core_count=1 if index < 8 else 2,
            config=_generator_config(),
        )
        for index in range(16)
    ]
    binary = []
    for index in range(16):
        if index < 4:
            structure = generate_binary_polar_holes(
                f"binary_gallery_{index:06d}",
                master_seed=20260806,
                hole_count=1,
                config=_generator_config(),
            )
        elif index < 8:
            structure = generate_binary_polar_holes(
                f"binary_gallery_{index:06d}",
                master_seed=20260806,
                hole_count=2,
                config=_generator_config(),
            )
        else:
            structure = generate_binary_spectral_threshold(
                f"binary_gallery_{index:06d}",
                master_seed=20260806,
                config=_generator_config(),
            )
        binary.append(structure)
    return graded, binary


def test_stratified_selection_is_reproducible_test_only_and_quota_balanced() -> None:
    graded, binary = _structures()
    test_indices = np.arange(16, dtype=np.int64)
    graded_quotas = {"graded_core_1": 8, "graded_core_2": 8}
    binary_quotas = {"polar_hole_1": 4, "polar_hole_2": 4, "spectral_threshold": 8}

    first = select_stratified_test_indices(graded, test_indices, lane="graded_medium", quotas=graded_quotas, seed=7)
    second = select_stratified_test_indices(graded, test_indices, lane="graded_medium", quotas=graded_quotas, seed=7)
    binary_selected = select_stratified_test_indices(
        binary, test_indices, lane="binary_air_holes", quotas=binary_quotas, seed=7
    )

    assert first == second
    assert set(first) <= set(test_indices.tolist())
    assert len(first) == len(set(first)) == 16
    assert [infer_full_cell_stratum(graded[index], "graded_medium") for index in first].count("graded_core_1") == 8
    strata = [infer_full_cell_stratum(binary[index], "binary_air_holes") for index in binary_selected]
    assert {name: strata.count(name) for name in binary_quotas} == binary_quotas


def test_sampled_gxm_path_is_exact_13_point_no_interpolation() -> None:
    _, grid_index = build_full_bz_grid(9)
    path, ticks, labels = sampled_gxm_path(grid_index)
    coordinates = grid_index[path].tolist()

    assert len(path) == 13
    assert ticks == (0, 4, 8, 12)
    assert labels == (r"$\Gamma$", r"$X_9$", r"$M_9$", r"$\Gamma$")
    assert coordinates[0] == [0, 0]
    assert coordinates[4] == [4, 0]
    assert coordinates[8] == [4, 4]
    assert coordinates[-1] == [0, 0]
    assert len(path) == len(coordinates)


def test_largest_gap_edge_selects_complete_and_pseudo_gap() -> None:
    complete = np.tile(np.arange(10, dtype=np.float64)[None, :] * 0.1, (81, 1))
    complete[:, 3] += 0.3
    complete[:, 4:] += 0.3
    selected = select_largest_gap_edge(complete)
    assert selected.lower_band_one_based == 3
    assert selected.upper_band_one_based == 4
    assert selected.status == "complete_gap"
    assert selected.gap == pytest.approx(0.4)

    pseudo = np.tile(np.arange(10, dtype=np.float64)[None, :] * 0.01, (81, 1))
    variation = np.linspace(0.0, 0.2, 81)[:, None]
    pseudo = pseudo + variation
    selected_pseudo = select_largest_gap_edge(pseudo)
    assert selected_pseudo.status == "pseudo_gap"
    assert selected_pseudo.gap < 0.0


def test_electric_energy_density_is_finite_nonnegative_and_normalized() -> None:
    efield = np.ones((8, 12, 1, 3), dtype=np.complex128) * (1.0 + 1.0j)
    hfield = np.ones_like(efield) * (0.5 - 0.2j)
    epsilon = np.linspace(1.0, 8.9, 24, dtype=np.float64).reshape(4, 6)

    e_normalized, h_normalized, energy, metadata = normalized_electric_energy_density(
        efield, hfield, epsilon=epsilon
    )

    assert e_normalized.shape == h_normalized.shape == efield.shape
    assert energy.shape == (8, 12)
    assert np.all(np.isfinite(energy))
    assert np.all(energy >= 0.0)
    assert float(np.max(energy)) == pytest.approx(1.0)
    assert metadata["visual_normalization"] == "per_sample_max_equals_one"


class _FakeVector3:
    def __init__(self, *values):
        self.values = values


class _FakeMedium:
    def __init__(self, *, epsilon):
        self.epsilon = epsilon


class _FakeLattice:
    def __init__(self, *, size):
        self.size = size


class _FakeModeSolver:
    latest = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.all_freqs = np.asarray([[0.8, 0.2, 0.5, 0.3, 0.4, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]])
        self.ran = None
        _FakeModeSolver.latest = self

    def run_te(self):
        self.ran = "TE"

    def get_efield(self, band, *, bloch_phase):
        assert bloch_phase is True
        return np.full((8, 8, 1, 3), complex(band, 1.0))

    def get_hfield(self, band, *, bloch_phase):
        assert bloch_phase is True
        return np.full((8, 8, 1, 3), complex(1.0, band))


class _FakeMp:
    __version__ = "1.33.0-beta"
    Vector3 = _FakeVector3
    Medium = _FakeMedium
    Lattice = _FakeLattice


class _FakeMpb:
    __version__ = (1, 12, 0)
    ModeSolver = _FakeModeSolver


def test_full_cell_eigenfield_preserves_canonical_to_solver_band_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    graded, _ = _structures()
    monkeypatch.setattr(full_cell_solver, "load_mpb_modules", lambda: (_FakeMp, _FakeMpb))
    config = FullCellMpbBandSolverConfig(
        epsilon_air=1.0,
        epsilon_material=8.9,
        resolution=64,
        polarization="TE",
        num_bands=10,
        band_guard_count=4,
        band_order_policy="frequency_ascending",
        eigensolver_tolerance=1.0e-7,
        mesh_size=3,
    )

    result = solve_eigenfield_for_full_cell_mpb(
        graded[0], k_point=(0.0, 0.0), band_index_one_based=3, config=config
    )

    assert result.eigenfrequency == pytest.approx(0.4)
    assert result.requested_band_index_one_based == 3
    assert result.solver_band_index_one_based == 5
    assert _FakeModeSolver.latest.ran == "TE"
    assert _FakeModeSolver.latest.kwargs["mesh_size"] == 3
    assert result.metadata["geometry_hash"] == graded[0].geometry_hash


def _write_gallery_inputs(tmp_path: Path) -> Path:
    pytest.importorskip("h5py")
    graded, binary = _structures()
    k_points, k_grid_index = build_full_bz_grid(9)
    bands = np.tile(np.arange(1, 11, dtype=np.float32)[None, None, :] * 0.1, (16, 81, 1))
    for name, structures, lane in (
        ("graded.h5", graded, "graded_medium"),
        ("binary.h5", binary, "binary_air_holes"),
    ):
        write_full_cell_hdf5(
            tmp_path / name,
            structures=structures,
            Wh=bands,
            k_points=k_points,
            k_grid_index=k_grid_index,
            idx_train=np.asarray([], dtype=np.int64),
            idx_val=np.asarray([], dtype=np.int64),
            idx_test=np.arange(16, dtype=np.int64),
            raster_resolutions=[128],
            metadata={"lane": lane},
        )
    config = {
        "experiment": {"experiment_id": "gallery_unit", "family": "research", "stage": "visualization"},
        "datasets": [
            {
                "path": str(tmp_path / "graded.h5"),
                "lane": "graded_medium",
                "status": "cluster_generated_pending_wsl_acceptance",
                "quotas": {"graded_core_1": 8, "graded_core_2": 8},
            },
            {
                "path": str(tmp_path / "binary.h5"),
                "lane": "binary_air_holes",
                "status": "wsl_cross_environment_acceptance_pass",
                "quotas": {"polar_hole_1": 4, "polar_hole_2": 4, "spectral_threshold": 8},
            },
        ],
        "sampling": {
            "split": "test",
            "sample_count_per_dataset": 16,
            "policy": "stratified_random_without_replacement",
            "seed": 20260806,
        },
        "bands": {"gap_lower_bands_one_based": [2, 9]},
        "field": {
            "quantity": "epsilon_abs_e_squared",
            "normalization": "per_sample_max",
            "frequency_mre_max": 0.01,
        },
        "solver": {
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
        "output": {"runs_dir": str(tmp_path / "runs"), "formats": ["png", "svg", "pdf"], "dpi": 72},
    }
    path = tmp_path / "gallery.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _fake_eigenfield_solver(structure, *, k_point, band_index_one_based, config):
    field = np.ones((8, 8, 1, 3), dtype=np.complex128)
    return MpbEigenfieldResult(
        eigenfrequency=float(band_index_one_based) * 0.1,
        efield=field,
        hfield=field * (0.2 + 0.1j),
        requested_band_index_one_based=int(band_index_one_based),
        solver_band_index_one_based=int(band_index_one_based),
        k_point=tuple(float(value) for value in k_point),
        metadata={"solver_backend": "deterministic_fake", "sample_id": structure.sample_id},
    )


def test_gallery_runner_writes_traceable_smoke_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    config = _write_gallery_inputs(tmp_path)
    run_dir = tmp_path / "gallery_run"

    out = build_full_cell_sample_gallery(
        config,
        run_dir=run_dir,
        max_samples_per_dataset=1,
        solver_fn=_fake_eigenfield_solver,
        skip_environment_check=True,
    )

    assert out == run_dir
    for name in (
        "run.log",
        "run.meta.json",
        "config.resolved.yaml",
        "results.json",
        "metrics.json",
        "history.csv",
        "selected_samples.csv",
        "band_source.csv",
        "near_field_records.h5",
        "figure_manifest.json",
    ):
        assert (run_dir / name).is_file(), name
    figures = sorted((run_dir / "figures").glob("*"))
    assert len(figures) == 18
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "figure_manifest.json").read_text(encoding="utf-8"))
    assert results["status"] == "full_cell_sample_gallery_complete"
    assert results["sample_count_total"] == 2
    assert results["graded_dataset_acceptance_advanced"] is False
    assert results["far_field_computed"] is False
    assert manifest["band_path"]["interpolation"] is False
    assert manifest["near_field"]["cross_sample_absolute_magnitude_comparable"] is False
    assert manifest["evidence_boundary"]["graded_medium"] == "cluster_generated_pending_wsl_acceptance"


def test_gallery_cli_help_does_not_import_mpb(monkeypatch: pytest.MonkeyPatch) -> None:
    from phc2d.cli import main

    monkeypatch.setitem(__import__("sys").modules, "meep", None)
    with pytest.raises(SystemExit) as exc:
        main(["plot", "full-cell-gallery", "--help"])
    assert exc.value.code == 0
