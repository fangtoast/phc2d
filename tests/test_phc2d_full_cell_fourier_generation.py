"""Data-contract tests for unified Fourier B/G generation and rasters.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from phc2d.data.full_cell import (
    MASTER_BLOCK_MEAN_V1,
    block_mean_downsample,
    build_full_cell_raster_pyramid,
    hash_raster_content,
    load_full_cell_hdf5,
    write_full_cell_hdf5,
    write_full_cell_hdf5_streaming,
)
from phc2d.data.full_cell_generation import (
    _structure_schedule,
    _validate_config,
    load_full_cell_generation_config,
)
from phc2d.generators.full_cell import (
    generate_unified_fourier_binary,
    generate_unified_fourier_graded,
)
from phc2d.generators.full_cell_latent import derive_full_cell_seed64


SMALL_CONFIG = {
    "fourier_cutoff": 3,
    "audit_resolution": 32,
    "binary_target_material_fraction_range": [0.50, 0.90],
    "graded_target_material_fraction_range": [0.60, 0.78],
    "graded_temperature_range": [0.08, 0.16],
    "target_material_fraction_tolerance": 0.01,
    "minimum_gray_fraction": 0.01,
}
ROOT = Path(__file__).resolve().parents[1]


def test_seed64_is_reproducible_and_not_truncated_to_32_bits() -> None:
    first = derive_full_cell_seed64(20260817, "sample", "binary")
    assert first == derive_full_cell_seed64(20260817, "sample", "binary")
    assert first != derive_full_cell_seed64(20260817, "sample", "graded")
    assert first > 2**32


def test_unified_binary_is_reproducible_strict_binary_and_periodic() -> None:
    first = generate_unified_fourier_binary(
        "binary_000000",
        master_seed=20260817,
        material_fraction_bin=2,
        config=SMALL_CONFIG,
    )
    second = generate_unified_fourier_binary(
        "binary_000000",
        master_seed=20260817,
        material_fraction_bin=2,
        config=SMALL_CONFIG,
    )
    assert first.parameter_hash == second.parameter_hash
    np.testing.assert_array_equal(first.rasterize((32, 32)), second.rasterize((32, 32)))
    assert set(np.unique(first.rasterize((32, 32)))) <= {0.0, 1.0}
    y = np.linspace(-0.5, 0.5, 17, endpoint=False)
    np.testing.assert_array_equal(
        first.evaluate_material_weight(-0.5, y),
        first.evaluate_material_weight(0.5, y),
    )


def test_unified_graded_hits_target_and_uses_requested_strata() -> None:
    spec = generate_unified_fourier_graded(
        "graded_000000",
        master_seed=20260818,
        material_fraction_bin=1,
        temperature_bin=3,
        config=SMALL_CONFIG,
    )
    weight = spec.rasterize((32, 32))
    assert np.all(np.isfinite(weight))
    assert np.min(weight) >= 0.0
    assert np.max(weight) <= 1.0
    assert np.mean(weight) == pytest.approx(
        spec.parameters["target_material_fraction"],
        abs=1.0e-5,
    )
    assert spec.metadata["material_fraction_bin"] == 1
    assert spec.metadata["temperature_bin"] == 3


def test_master_block_mean_uses_direct_master_for_every_resolution() -> None:
    master = np.arange(64, dtype=np.float32).reshape(8, 8) / 63.0
    expected = master.reshape(2, 4, 2, 4).mean(axis=(1, 3))
    np.testing.assert_array_equal(block_mean_downsample(master, 2), expected)

    spec = generate_unified_fourier_binary(
        "binary_000001",
        master_seed=20260817,
        material_fraction_bin=0,
        config=SMALL_CONFIG,
    )
    pyramid = build_full_cell_raster_pyramid(
        spec,
        resolutions=[8, 16, 32],
        source_resolution=32,
        method=MASTER_BLOCK_MEAN_V1,
    )
    np.testing.assert_array_equal(
        pyramid.grids[8],
        block_mean_downsample(pyramid.grids[32], 8),
    )
    assert np.any((pyramid.grids[8] > 0.0) & (pyramid.grids[8] < 1.0))


def test_content_hash_ignores_identity_and_respects_quantization() -> None:
    binary = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    changed = binary.copy()
    changed[0, 0] = 1
    assert hash_raster_content(binary, material_kind="binary") != hash_raster_content(
        changed,
        material_kind="binary",
    )
    graded = np.asarray([[0.25, 0.75]], dtype=np.float64)
    within_bin = graded + 1.0e-8
    across_bin = graded + 1.0e-3
    assert hash_raster_content(graded, material_kind="continuous") == hash_raster_content(
        within_bin,
        material_kind="continuous",
    )
    assert hash_raster_content(graded, material_kind="continuous") != hash_raster_content(
        across_bin,
        material_kind="continuous",
    )


def test_v2_binary_hdf5_uses_semantic_fields_and_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    structures = [
        generate_unified_fourier_binary(
            f"binary_{index:06d}",
            master_seed=20260817,
            material_fraction_bin=index,
            config=SMALL_CONFIG,
        )
        for index in range(2)
    ]
    path = write_full_cell_hdf5(
        tmp_path / "v2.h5",
        structures=structures,
        Wh=np.zeros((2, 9, 3), dtype=np.float32),
        k_points=np.zeros((9, 2), dtype=np.float64),
        k_grid_index=np.zeros((9, 2), dtype=np.int64),
        idx_train=np.asarray([0], dtype=np.int64),
        idx_val=np.asarray([1], dtype=np.int64),
        idx_test=np.asarray([], dtype=np.int64),
        raster_resolutions=[8, 16, 32],
        raster_method=MASTER_BLOCK_MEAN_V1,
        raster_source_resolution=32,
        metadata={"protocol": "full_cell_fourier_v2"},
    )
    bundle = load_full_cell_hdf5(path)
    assert set(bundle.material_binary_grids) == {32}
    assert set(bundle.material_area_fraction_grids) == {8, 16}
    assert bundle.material_weight_grids == {}
    assert bundle.epsilon_grids == {}
    assert set(bundle.content_hashes) == {8, 16, 32}
    assert bundle.structures[0].parameter_hash == structures[0].parameter_hash


def test_v2_streaming_writer_matches_semantic_raster_contract(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    structure = generate_unified_fourier_graded(
        "graded_000002",
        master_seed=20260818,
        material_fraction_bin=2,
        temperature_bin=1,
        config=SMALL_CONFIG,
    )
    path = write_full_cell_hdf5_streaming(
        tmp_path / "streaming_v2.h5",
        structures=[structure],
        band_shape=(9, 3),
        band_loader=lambda _index: np.zeros((9, 3), dtype=np.float32),
        k_points=np.zeros((9, 2), dtype=np.float64),
        k_grid_index=np.zeros((9, 2), dtype=np.int64),
        idx_train=np.asarray([0], dtype=np.int64),
        idx_val=np.asarray([], dtype=np.int64),
        idx_test=np.asarray([], dtype=np.int64),
        raster_resolutions=[8, 16, 32],
        raster_method=MASTER_BLOCK_MEAN_V1,
        raster_source_resolution=32,
        metadata={"protocol": "full_cell_fourier_v2"},
        raster_workers=1,
    )
    bundle = load_full_cell_hdf5(path)
    assert set(bundle.material_weight_grids) == {8, 16, 32}
    assert set(bundle.content_hashes) == {8, 16, 32}
    np.testing.assert_array_equal(
        bundle.material_weight_grids[8][0],
        block_mean_downsample(bundle.material_weight_grids[32][0], 8),
    )


def test_pilot_configs_freeze_64_rows_and_resolution() -> None:
    config_paths = [
        ROOT / "configs/research/full_cell_binary_fourier_bz9_64_xingyi_pilot_v1.yaml",
        ROOT / "configs/research/full_cell_graded_fourier_bz9_64_xingyi_pilot_v1.yaml",
    ]
    for path in config_paths:
        cfg = load_full_cell_generation_config(path)
        assert _validate_config(cfg) == []
        assert cfg["dataset"]["sample_count"] == 64
        assert cfg["dataset"]["split"] == {"train": 48, "val": 8, "test": 8}
        assert cfg["solver"]["resolution_status"] == "frozen_resolution_64"
        assert cfg["solver"]["resolution_evidence"].endswith("全胞MPB分辨率收敛验证_20260817.md")
        assert cfg["dataset"]["raster_resolutions"] == [32, 64, 128, 256]

    binary = _structure_schedule("binary_fourier", 64)
    assert {row[2] for row in binary} == {
        "binary_material_q1",
        "binary_material_q2",
        "binary_material_q3",
        "binary_material_q4",
    }
    assert all(sum(row[2] == stratum for row in binary) == 16 for stratum in {row[2] for row in binary})
    graded = _structure_schedule("graded_fourier", 64)
    assert len({row[2] for row in graded}) == 16
    assert all(sum(row[2] == stratum for row in graded) == 4 for stratum in {row[2] for row in graded})


def test_xingyi_fourier_jobs_are_dry_run_first_and_fail_closed() -> None:
    compute = (ROOT / "jobs/xingyi_full_cell_fourier_pilot.sh").read_text(encoding="utf-8")
    submit = (ROOT / "jobs/xingyi_submit_full_cell_fourier_pilot.sh").read_text(encoding="utf-8")
    assert "PHC2D_AUTHORIZE_FOURIER_SMOKE" in compute
    assert "PHC2D_AUTHORIZE_FOURIER_PILOT" in compute
    assert "provisional" not in compute.lower()
    assert 'PHC2D_SUBMIT_MODE="${PHC2D_SUBMIT_MODE:-dry-run}"' in submit
    assert "PHC2D_SUBMIT_MODE=submit" not in submit
    assert "full_cell_binary_fourier_bz9_64_xingyi_pilot_v1.yaml" in compute
    assert "full_cell_graded_fourier_bz9_64_xingyi_pilot_v1.yaml" in compute


def test_paired_fourier_cutoffs_share_exact_mode_coefficients() -> None:
    base = {
        **SMALL_CONFIG,
        "paired_coefficient_seed": 2026081851,
        "fixed_target_material_fraction": 0.70,
    }
    k5 = generate_unified_fourier_binary(
        "paired", master_seed=1, material_fraction_bin=0,
        config={**base, "fourier_cutoff": 5},
    )
    k20 = generate_unified_fourier_binary(
        "paired", master_seed=1, material_fraction_bin=0,
        config={**base, "fourier_cutoff": 20},
    )
    modes5 = {(row["mx"], row["my"]): (row["cos"], row["sin"]) for row in k5.parameters["modes"]}
    modes20 = {(row["mx"], row["my"]): (row["cos"], row["sin"]) for row in k20.parameters["modes"]}
    assert modes5 and set(modes5) < set(modes20)
    assert all(modes20[key] == value for key, value in modes5.items())
    assert k5.parameters["target_material_fraction"] == pytest.approx(0.70)
    assert k20.parameters["target_material_fraction"] == pytest.approx(0.70)


def test_cutoff_study_configs_are_single_sample_frozen_r64() -> None:
    for lane in ("binary", "graded"):
        path = ROOT / f"configs/research/full_cell_{lane}_fourier_cutoff_pair1_xingyi_v1.yaml"
        cfg = load_full_cell_generation_config(path)
        assert _validate_config(cfg) == []
        assert cfg["dataset"]["sample_count"] == 1
        assert cfg["dataset"]["split"] == {"train": 1, "val": 0, "test": 0}
        assert cfg["solver"]["resolution"] == 64
        assert cfg["generator"]["paired_coefficient_seed"]
