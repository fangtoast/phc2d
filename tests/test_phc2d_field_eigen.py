"""Tests for Bloch-eigenfield mode lineage, normalization, and serialization.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util
from collections import Counter

import numpy as np
import pytest

from phc2d.data import FieldEigenRecord, build_field_eigen_operator_selection_manifest, build_field_eigen_selection_manifest, canonicalize_eigenfields, inspect_field_dataset, validate_field_eigen_selection_manifest, write_field_eigen_hdf5
from phc2d.data.k_path import KPathMetadata, build_square_gxm_reduced_spec, validate_verified_k_path_reference
from phc2d.solvers import MpbBandSolverConfig, solve_eigenfield_bundle_for_rho_mpb, solve_eigenfield_for_rho_mpb, solve_eigenfield_multiband_bundle_for_rho_mpb
import phc2d.solvers.mpb as mpb_module


class _FakeMp:
    inf = float("inf")
    __version__ = "fake-mpb"

    @staticmethod
    def Vector3(*values):
        return tuple(values)

    @staticmethod
    def Medium(*, epsilon):
        return {"epsilon": epsilon}

    @staticmethod
    def Block(**kwargs):
        return kwargs

    @staticmethod
    def Lattice(*, size):
        return {"size": size}


class _FakeModeSolver:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.all_freqs = np.asarray([[0.31, 0.11, 0.21]], dtype=np.float64)

    def run_te(self):
        return None

    def run_tm(self):
        return None

    def get_efield(self, band, *, bloch_phase):
        assert bloch_phase is True
        return np.full((4, 5, 1, 3), complex(band, 1.0), dtype=np.complex128)

    def get_hfield(self, band, *, bloch_phase):
        assert bloch_phase is True
        return np.full((4, 5, 1, 3), complex(2 * band, -1.0), dtype=np.complex128)


class _FakeMpb:
    ModeSolver = _FakeModeSolver


def _solver_config() -> MpbBandSolverConfig:
    return MpbBandSolverConfig(
        num_bands=3,
        resolution=8,
        polarization="TE",
        epsilon_air=1.0,
        epsilon_material=8.9,
        eigensolver_tolerance=1.0e-7,
        grid_shape=(4, 4),
    )


def test_eigenfield_solver_preserves_canonical_to_raw_band_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mpb_module, "load_mpb_modules", lambda: (_FakeMp, _FakeMpb))

    result = solve_eigenfield_for_rho_mpb(
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        k_point=(0.1, 0.2),
        band_index_one_based=1,
        config=_solver_config(),
    )

    assert result.eigenfrequency == pytest.approx(0.11)
    assert result.solver_band_index_one_based == 2
    assert result.metadata["requested_band_index_one_based"] == 1
    assert result.metadata["solver_band_index_one_based"] == 2
    assert result.metadata["material_parameters"] == {"epsilon_air": 1.0, "epsilon_material": 8.9}
    assert result.efield.shape == (4, 5, 1, 3)


def test_eigenfield_worker_bundle_preserves_both_complex_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mpb_module, "load_mpb_modules", lambda: (_FakeMp, _FakeMpb))
    bundle, metadata = solve_eigenfield_bundle_for_rho_mpb(
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        k_points=[(0.1, 0.2)],
        config=_solver_config().__class__(**{**_solver_config().__dict__, "field_band_index_one_based": 1}),
    )

    assert bundle.shape == (2, 4, 5, 1, 3)
    assert np.iscomplexobj(bundle)
    assert metadata["eigenfrequency"] == pytest.approx(0.11)


def test_multiband_worker_solves_once_and_preserves_band_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mpb_module, "load_mpb_modules", lambda: (_FakeMp, _FakeMpb))
    config = _solver_config().__class__(**{**_solver_config().__dict__, "field_band_indices_one_based": (1, 2, 3)})
    bundle, metadata = solve_eigenfield_multiband_bundle_for_rho_mpb(
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        k_points=[(0.1, 0.2)],
        config=config,
    )
    assert bundle.shape == (3, 2, 4, 5, 1, 3)
    assert metadata["eigenfrequencies"] == pytest.approx([0.11, 0.21, 0.31])
    assert metadata["solver_band_indices_one_based"] == [2, 3, 1]


def test_canonicalize_eigenfields_uses_one_scale_and_phase_for_e_and_h() -> None:
    efield = np.asarray([[[1.0 + 1.0j], [2.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]])
    hfield = 3.0 * efield
    epsilon = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    normalized_e, normalized_h, metadata = canonicalize_eigenfields(
        efield, hfield, epsilon=epsilon, domain_extent=(2.0, 1.0)
    )

    weighted_norm = np.sum(epsilon[..., None] * np.abs(normalized_e) ** 2) * metadata["dx"] * metadata["dy"]
    assert weighted_norm == pytest.approx(1.0)
    assert metadata["electric_inner_product"] == "integral_conjugate_e1_dot_epsilon_e2_dv"
    assert normalized_e.reshape(-1)[metadata["anchor_flat_index"]].imag == pytest.approx(0.0, abs=1.0e-12)
    assert normalized_e.reshape(-1)[metadata["anchor_flat_index"]].real >= 0.0
    assert np.allclose(normalized_h, 3.0 * normalized_e)


def test_canonicalization_metadata_preserves_anchor_across_tied_maxima() -> None:
    efield = np.zeros((2, 2, 3), dtype=np.complex128)
    hfield = np.zeros_like(efield)
    efield.reshape(-1)[0] = 1.0j
    efield.reshape(-1)[1] = 1.0
    hfield.reshape(-1)[0] = 2.0j

    normalized_e, _, metadata = canonicalize_eigenfields(
        efield, hfield, epsilon=np.ones((2, 2)), domain_extent=(1.0, 1.0)
    )
    stored = normalized_e.astype(np.complex64)
    anchor = stored.reshape(-1)[int(metadata["anchor_flat_index"])]

    assert abs(float(anchor.imag)) < 1.0e-6
    assert float(anchor.real) >= 0.0


@pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py is optional")
def test_field_eigen_writer_keeps_native_complex_aliases_and_contract_metadata(tmp_path) -> None:
    efield, hfield, normalization = canonicalize_eigenfields(
        np.ones((3, 4, 1, 3), dtype=np.complex128),
        np.full((3, 4, 1, 3), 2.0 + 0.0j, dtype=np.complex128),
        epsilon=np.ones((3, 4)),
        domain_extent=(1.0, 1.0),
    )
    record = FieldEigenRecord(
        record_id="record_000",
        source_sample_id="source_000",
        source_sample_hash="hash",
        source_split="train",
        rho=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        epsilon=np.ones((4, 4), dtype=np.float32),
        k_point=np.asarray([0.1, 0.2]),
        k_path_id="square_gxm",
        band_index_one_based=2,
        eigenfrequency=0.25,
        efield_complex=efield,
        hfield_complex=hfield,
        polarization="TE",
        bloch_phase=np.asarray([0.1, 0.2]),
        lattice_constant=1.0,
        domain_extent=np.asarray([1.0, 1.0]),
        dx=normalization["dx"],
        dy=normalization["dy"],
        field_normalization=normalization,
        solver_provenance={"solver_backend": "fake"},
    )
    path = write_field_eigen_hdf5(tmp_path / "field.h5", [record], metadata={"boundary_condition": "bloch_periodic"})
    schema = inspect_field_dataset(path)

    assert schema.has_field("efield")
    assert schema.has_field("hfield")
    assert "efield_complex" in schema.fields
    assert "hfield_complex" in schema.fields
    assert schema.boundary_condition == "bloch_periodic"
    assert schema.has_field("source_split")


def test_verified_k_path_reference_requires_exact_coordinate_equivalence() -> None:
    source = build_square_gxm_reduced_spec(
        n_total=7,
        k_path_id="test_square_gxm",
        metadata=KPathMetadata(verification_status="shards_merged"),
    )
    reference = build_square_gxm_reduced_spec(
        n_total=7,
        k_path_id="test_square_gxm",
        metadata=KPathMetadata(verification_status="physically_verified_with_documented_numerical_tolerance"),
    )

    proof = validate_verified_k_path_reference(source, reference)

    assert proof["coordinate_verification"] == "inherited_from_exact_verified_reference"
    assert proof["source_k_points_hash"] == proof["reference_k_points_hash"]


def test_field_selection_manifest_is_deterministic_balanced_and_split_safe() -> None:
    rho = np.asarray([[float(index % 8), 1.0] for index in range(80)], dtype=np.float32)
    rows = [{"sample_id": f"sample_{index:03d}", "rho_hash": f"rho_{index}", "geometry_hash": f"geo_{index}"} for index in range(80)]
    k_spec = build_square_gxm_reduced_spec(n_total=13, k_path_id="field_selection_unit")
    kwargs = {
        "rho": rho,
        "metadata_rows": rows,
        "split_indices": {"train": np.arange(0, 56), "val": np.arange(56, 68), "test": np.arange(68, 80)},
        "k_spec": k_spec,
        "dataset_path": "unit.npz",
        "dataset_sha256": "unit-hash",
        "contract_id": "field-contract",
        "contract_version": 2,
        "split_counts": {"train": 48, "val": 8, "test": 8},
        "n_occupancy_strata": 4,
        "n_k_strata": 8,
        "band_range_one_based": (2, 10),
        "selector_seed": 20260720,
    }
    first = build_field_eigen_selection_manifest(**kwargs)
    second = build_field_eigen_selection_manifest(**kwargs)

    assert first == second
    assert validate_field_eigen_selection_manifest(first) == []
    assert len(first["records"]) == 64
    assert len({row["source_sample_index"] for row in first["records"]}) == 64
    assert {name: sum(row["source_split"] == name for row in first["records"]) for name in ("train", "val", "test")} == {"train": 48, "val": 8, "test": 8}
    counts = [sum(row["band_index_one_based"] == band for row in first["records"]) for band in range(2, 11)]
    assert max(counts) - min(counts) <= 1
    assert len({row["physical_k_id"] for row in first["records"]}) == 8


def test_field_selection_supports_1000_unique_stratified_records() -> None:
    rho = np.asarray([[float(index % 11), 1.0] for index in range(1400)], dtype=np.float32)
    rows = [{"sample_id": f"sample_{index:04d}", "rho_hash": f"rho_{index}", "geometry_hash": f"geo_{index}"} for index in range(1400)]
    manifest = build_field_eigen_selection_manifest(
        rho=rho, metadata_rows=rows,
        split_indices={"train": np.arange(0, 1000), "val": np.arange(1000, 1200), "test": np.arange(1200, 1400)},
        k_spec=build_square_gxm_reduced_spec(n_total=17, k_path_id="field_selection_1000_unit"),
        dataset_path="unit.npz", dataset_sha256="unit-hash", contract_id="field-contract", contract_version=3,
        split_counts={"train": 750, "val": 125, "test": 125}, n_occupancy_strata=4, n_k_strata=8,
        band_range_one_based=(2, 10), selector_seed=20260720,
    )
    assert manifest["selection_id"] == "field_eigen_te_1000_v1"
    assert validate_field_eigen_selection_manifest(manifest) == []
    assert len({row["source_sample_index"] for row in manifest["records"]}) == 1000
    assert {name: sum(row["source_split"] == name for row in manifest["records"]) for name in ("train", "val", "test")} == {"train": 750, "val": 125, "test": 125}


def test_operator_field_selection_has_repeated_structure_k_and_balanced_bands() -> None:
    rho = np.asarray([[float(index % 11), 1.0] for index in range(400)], dtype=np.float32)
    rows = [{"sample_id": f"sample_{index:04d}", "rho_hash": f"rho_{index}", "geometry_hash": f"geo_{index}"} for index in range(400)]
    kwargs = dict(
        rho=rho,
        metadata_rows=rows,
        split_indices={"train": np.arange(0, 300), "val": np.arange(300, 350), "test": np.arange(350, 400)},
        k_spec=build_square_gxm_reduced_spec(n_total=17, k_path_id="field_operator_unit"),
        dataset_path="unit.npz",
        dataset_sha256="unit-hash",
        contract_id="field-contract",
        contract_version=3,
        structure_split_counts={"train": 192, "val": 32, "test": 32},
        n_occupancy_strata=4,
        n_k_strata=8,
        k_per_structure=4,
        band_range_one_based=(2, 10),
        selector_seed=20260720,
    )
    first = build_field_eigen_operator_selection_manifest(**kwargs)
    second = build_field_eigen_operator_selection_manifest(**kwargs)
    assert first == second
    assert validate_field_eigen_selection_manifest(first) == []
    records = first["records"]
    assert len(records) == 9216
    assert len(first["solver_jobs"]) == 1024
    assert len({row["source_sample_index"] for row in records}) == 256
    assert set(Counter(row["source_sample_index"] for row in records).values()) == {36}
    assert {band: sum(row["band_index_one_based"] == band for row in records) for band in range(2, 11)} == {band: 1024 for band in range(2, 11)}
    assert {split: sum(row["source_split"] == split for row in records) for split in ("train", "val", "test")} == {"train": 6912, "val": 1152, "test": 1152}
