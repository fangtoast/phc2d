"""Tests for non-p4m full-cell structure and HDF5 contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phc2d.data import (
    DataRepresentationSpec,
    build_full_bz_grid,
    infer_dataset_schema,
    load_full_cell_hdf5,
    make_input,
    validate_representation_compatibility,
    write_full_cell_hdf5,
)
from phc2d.data.full_cell_generation import _validate_config, load_full_cell_generation_config
from phc2d.generators import (
    generate_binary_polar_holes,
    generate_binary_spectral_threshold,
    generate_graded_rbf_fourier,
)


def _generator_config() -> dict[str, float | int]:
    return {
        "epsilon_air": 1.0,
        "epsilon_material": 8.9,
        "audit_resolution": 64,
        "fourier_cutoff": 4,
        "max_attempts": 128,
    }


def test_full_bz_grid_is_centered_square_and_duplicate_free() -> None:
    points, indices = build_full_bz_grid(9)

    assert points.shape == (81, 2)
    assert indices.shape == (81, 2)
    assert len(np.unique(points, axis=0)) == 81
    assert np.any(np.all(indices == (0, 0), axis=1))
    assert np.allclose(points, indices / 9.0)
    assert np.max(np.abs(points)) == pytest.approx(4.0 / 9.0)
    with pytest.raises(ValueError, match="odd"):
        build_full_bz_grid(8)


def test_graded_generator_is_deterministic_continuous_and_periodic() -> None:
    first = generate_graded_rbf_fourier(
        "graded_unit_000000",
        master_seed=20260803,
        core_count=2,
        config=_generator_config(),
    )
    second = generate_graded_rbf_fourier(
        "graded_unit_000000",
        master_seed=20260803,
        core_count=2,
        config=_generator_config(),
    )
    grid = first.rasterize((48, 48))

    assert first.geometry_hash == second.geometry_hash
    assert first.material_kind == "continuous"
    assert np.all((grid >= 0.0) & (grid <= 1.0))
    assert np.mean((grid > 0.05) & (grid < 0.95)) >= 0.20
    assert 0.59 <= first.metadata["stats"]["material_fraction"] <= 0.79
    assert first.metadata["stats"]["dense_material_fraction"] >= 0.35
    assert first.metadata["stats"]["near_air_fraction"] >= 0.01
    assert first.metadata["stats"]["minimum"] <= 0.05
    assert first.metadata["stats"]["maximum"] >= 0.90
    assert all(float(blob["amplitude"]) < 0.0 for blob in first.parameters["blobs"])
    assert first.metadata["stats"]["p4m_symmetry_error"] > 1.0e-3
    probes = np.linspace(-0.49, 0.49, 17)
    assert np.allclose(
        first.evaluate_material_weight(-0.5, probes),
        first.evaluate_material_weight(0.5, probes),
        atol=1.0e-10,
    )


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (generate_binary_polar_holes, {"hole_count": 1}),
        (generate_binary_polar_holes, {"hole_count": 2}),
        (generate_binary_spectral_threshold, {}),
    ],
)
def test_binary_generators_are_strictly_quantized(factory, kwargs) -> None:
    structure = factory(
        f"binary_unit_{factory.__name__}_{kwargs.get('hole_count', 0)}",
        master_seed=20260804,
        config=_generator_config(),
        **kwargs,
    )
    grid = structure.rasterize((64, 64))

    assert structure.material_kind == "binary"
    assert set(np.unique(grid).tolist()) <= {0.0, 1.0}
    assert 0.07 <= float(np.mean(grid == 0.0)) <= 0.53
    assert structure.metadata["stats"]["p4m_symmetry_error"] > 1.0e-3


def test_spectral_generator_uses_configured_higher_cutoff_scale() -> None:
    config = {
        **_generator_config(),
        "fourier_cutoff": 8,
        "spectral_cutoff_scale_range": [2.5, 4.5],
    }
    structure = generate_binary_spectral_threshold(
        "binary_high_cutoff_unit",
        master_seed=20260804,
        config=config,
    )

    assert 2.5 <= float(structure.parameters["cutoff_scale"]) <= 4.5
    assert max(max(abs(int(mode["mx"])), abs(int(mode["my"]))) for mode in structure.parameters["modes"]) == 8


@pytest.mark.parametrize(
    "name,lane",
    [
        ("full_cell_graded_bz9_10000_v1.yaml", "graded_medium"),
        ("full_cell_binary_bz9_10000_v1.yaml", "binary_air_holes"),
    ],
)
def test_full_cell_10000_configs_are_valid_and_have_fixed_splits(name: str, lane: str) -> None:
    path = Path(__file__).parents[1] / "configs" / "research" / name
    config = load_full_cell_generation_config(path)

    assert _validate_config(config) == []
    assert config["dataset"]["lane"] == lane
    assert config["dataset"]["sample_count"] == 10_000
    assert config["dataset"]["split"] == {"train": 8_000, "val": 1_000, "test": 1_000}


def test_full_cell_hdf5_roundtrip_and_schema(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    structures = [
        generate_graded_rbf_fourier(
            "graded_h5_000000",
            master_seed=7,
            core_count=1,
            config=_generator_config(),
        ),
        generate_graded_rbf_fourier(
            "graded_h5_000001",
            master_seed=7,
            core_count=2,
            config=_generator_config(),
        ),
    ]
    k_points, k_grid_index = build_full_bz_grid(3)
    bands = np.arange(2 * 9 * 3, dtype=np.float32).reshape(2, 9, 3) / 100.0
    path = tmp_path / "full_cell.h5"
    write_full_cell_hdf5(
        path,
        structures=structures,
        Wh=bands,
        k_points=k_points,
        k_grid_index=k_grid_index,
        idx_train=np.asarray([0]),
        idx_val=np.asarray([1]),
        idx_test=np.asarray([], dtype=np.int64),
        raster_resolutions=[16, 32],
        metadata={"lane": "graded_medium"},
    )

    loaded = load_full_cell_hdf5(path)
    schema = infer_dataset_schema(path)
    representation = DataRepresentationSpec(
        input_repr="epsilon_grid",
        output_target="Wh",
        grid_shape=(32, 32),
        k_points=9,
        n_bands=3,
    )

    assert loaded.Wh.shape == (2, 9, 3)
    assert loaded.epsilon_grids[32].shape == (2, 32, 32)
    assert loaded.structures[0].geometry_hash == structures[0].geometry_hash
    assert schema["kind"] == "hdf5_full_cell_band"
    assert schema["grid_resolutions"] == [16, 32]
    assert schema["n_samples"] == 2
    assert validate_representation_compatibility(schema, representation) == []
    selected = make_input({"epsilon_grid_32": loaded.epsilon_grids[32][0]}, representation)
    assert selected.shape == (32, 32)
