"""Tests for train-only full-cell analytic inverse codecs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phc2d.data import build_full_bz_grid, write_full_cell_hdf5
from phc2d.generators import (
    generate_binary_polar_holes,
    generate_binary_spectral_threshold,
    generate_graded_rbf_fourier,
)
from phc2d.inverse.full_cell_codec import (
    FULL_CELL_VARIANTS,
    PRODUCTION_PHYSICAL_WIDTHS,
    FullCellAnalyticCodec,
    fit_full_cell_analytic_codecs,
    fit_full_cell_analytic_codecs_from_hdf5,
    full_cell_variant_id,
    load_full_cell_analytic_codecs,
    save_full_cell_analytic_codecs,
)


def _config(*, cutoff: int) -> dict[str, float | int | list[float]]:
    return {
        "epsilon_air": 1.0,
        "epsilon_material": 8.9,
        "audit_resolution": 64,
        "fourier_cutoff": cutoff,
        "max_attempts": 512,
        "spectral_cutoff_scale_range": [2.5, 4.5],
    }


@pytest.fixture(scope="module")
def production_structures():
    rows = []
    for seed in range(3):
        rows.extend(
            [
                generate_binary_polar_holes(
                    f"polar1_{seed}", master_seed=101 + seed, hole_count=1, config=_config(cutoff=8)
                ),
                generate_binary_polar_holes(
                    f"polar2_{seed}", master_seed=201 + seed, hole_count=2, config=_config(cutoff=8)
                ),
                generate_binary_spectral_threshold(
                    f"spectral_{seed}", master_seed=301 + seed, config=_config(cutoff=8)
                ),
                generate_graded_rbf_fourier(
                    f"graded1_{seed}", master_seed=401 + seed, core_count=1, config=_config(cutoff=3)
                ),
                generate_graded_rbf_fourier(
                    f"graded2_{seed}", master_seed=501 + seed, core_count=2, config=_config(cutoff=3)
                ),
            ]
        )
    return rows


def test_variant_identity_and_production_widths(production_structures) -> None:
    codecs = fit_full_cell_analytic_codecs(production_structures)

    assert tuple(codecs) == FULL_CELL_VARIANTS
    assert {key: codec.physical_width for key, codec in codecs.items()} == PRODUCTION_PHYSICAL_WIDTHS
    assert codecs["binary_polar_holes_1"].encoded_width == 13
    assert codecs["binary_polar_holes_2"].encoded_width == 26
    assert codecs["binary_spectral_threshold"].encoded_width == 289
    assert codecs["graded_rbf_fourier_1"].encoded_width == 60
    assert codecs["graded_rbf_fourier_2"].encoded_width == 69


def test_encode_decode_roundtrip_preserves_hard_geometry(production_structures) -> None:
    grouped = {variant: [] for variant in FULL_CELL_VARIANTS}
    for structure in production_structures:
        grouped[full_cell_variant_id(structure)].append(structure)

    for variant, structures in grouped.items():
        codec = FullCellAnalyticCodec.fit(structures)
        original = structures[-1]
        encoded = codec.encode_structure(original)
        decoded = codec.decode_structure(encoded, template=original)

        assert encoded.shape == (codec.encoded_width,)
        assert np.all(np.isfinite(encoded))
        assert decoded.geometry_hash == original.geometry_hash
        assert np.array_equal(decoded.rasterize((64, 64)), original.rasterize((64, 64))), variant
        assert decoded.parameter_names == original.parameter_names
        assert decoded.parameter_mask == original.parameter_mask


def test_codec_manifest_roundtrip(production_structures, tmp_path: Path) -> None:
    codecs = fit_full_cell_analytic_codecs(production_structures)
    path = save_full_cell_analytic_codecs(codecs, tmp_path / "codec.resolved.json")
    restored = load_full_cell_analytic_codecs(path)

    assert tuple(restored) == FULL_CELL_VARIANTS
    for variant in FULL_CELL_VARIANTS:
        assert restored[variant].to_mapping() == codecs[variant].to_mapping()


@pytest.mark.parametrize("variant", FULL_CELL_VARIANTS)
def test_soft_decoder_is_differentiable_and_tracks_hard_geometry(
    production_structures,
    variant: str,
) -> None:
    torch = pytest.importorskip("torch")
    structures = [item for item in production_structures if full_cell_variant_id(item) == variant]
    codec = FullCellAnalyticCodec.fit(structures)
    encoded = torch.tensor(codec.encode_structure(structures[0]), dtype=torch.float64, requires_grad=True)
    soft = codec.decode_soft_raster_tensor(
        encoded,
        grid_shape=(48, 48),
        binary_temperature=0.002,
        torch=torch,
    )
    hard = torch.as_tensor(structures[0].rasterize((48, 48)), dtype=torch.float64)
    weights = torch.linspace(0.5, 1.5, soft.numel(), dtype=soft.dtype).reshape_as(soft)
    loss = torch.mean(torch.square(soft - 0.25) * weights)
    loss.backward()

    assert soft.shape == (48, 48)
    assert torch.all(torch.isfinite(soft))
    assert encoded.grad is not None
    assert torch.all(torch.isfinite(encoded.grad))
    assert float(torch.linalg.vector_norm(encoded.grad)) > 0.0
    tolerance = 5.0e-8 if variant.startswith("graded_") else 0.08
    assert float(torch.mean(torch.abs(soft.detach() - hard))) <= tolerance


def test_hdf5_fitter_uses_only_idx_train(production_structures, tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    train = list(production_structures[:10])
    validation = list(production_structures[10:])
    structures = train + validation
    points, grid_indices = build_full_bz_grid(3)
    wh = np.arange(len(structures) * 9 * 2, dtype=np.float32).reshape(len(structures), 9, 2)
    path = write_full_cell_hdf5(
        tmp_path / "full_cell_codec.h5",
        structures=structures,
        Wh=wh,
        k_points=points,
        k_grid_index=grid_indices,
        idx_train=np.arange(len(train), dtype=np.int64),
        idx_val=np.arange(len(train), len(structures), dtype=np.int64),
        idx_test=np.asarray([], dtype=np.int64),
        raster_resolutions=[8],
        metadata={"purpose": "codec_test"},
    )

    expected = fit_full_cell_analytic_codecs(train)
    actual = fit_full_cell_analytic_codecs_from_hdf5(path)

    assert {key: value.to_mapping() for key, value in actual.items()} == {
        key: value.to_mapping() for key, value in expected.items()
    }


def test_codec_rejects_invalid_or_cross_variant_inputs(production_structures) -> None:
    polar = [item for item in production_structures if full_cell_variant_id(item) == "binary_polar_holes_1"]
    graded = next(item for item in production_structures if full_cell_variant_id(item) == "graded_rbf_fourier_1")
    codec = FullCellAnalyticCodec.fit(polar)

    with pytest.raises(ValueError, match="Expected variant"):
        codec.encode_structure(graded)
    with pytest.raises(ValueError, match="finite with width"):
        codec.decode_structure(np.zeros(codec.encoded_width - 1))
    bad_pair = codec.encode_structure(polar[0])
    bad_pair[:2] = 0.0
    with pytest.raises(ValueError, match="Degenerate sin/cos pair"):
        codec.decode_structure(bad_pair, template=polar[0])
    out_of_range = codec.encode_structure(polar[0])
    out_of_range[-1] = 1.0e6
    with pytest.raises(ValueError, match="outside the frozen"):
        codec.decode_structure(out_of_range, template=polar[0])
