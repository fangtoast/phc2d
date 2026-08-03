"""Tests for Gate 1 data representation protocol.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from phc2d.data import (
    CORE_FIELD_DATASETS,
    DataRepresentationSpec,
    OPERATOR_CONTEXT_FIELDS,
    TargetSpec,
    infer_dataset_schema,
    inspect_field_dataset,
    make_input,
    make_target,
    validate_representation_compatibility,
    wedge_indices,
)


def test_npz_schema_inspection_and_target_specs(tmp_path):
    grid_shape = (8, 8)
    n_wedge = len(wedge_indices(grid_shape))
    n_samples = 3
    n_k = 4
    n_bands = 2
    path = tmp_path / "forward.npz"
    np.savez(
        path,
        rho=np.zeros((n_samples, n_wedge), dtype=np.float32),
        Wh=np.ones((n_samples, n_k, n_bands), dtype=np.float32),
        idx_train=np.array([0, 1], dtype=np.int64),
        idx_test=np.array([2], dtype=np.int64),
    )

    schema = infer_dataset_schema(path)
    assert schema["n_samples"] == n_samples
    assert schema["rho_wedge_length"] == n_wedge
    assert schema["k_points"] == n_k
    assert schema["n_bands"] == n_bands
    assert schema["splits"]["idx_train"] == 2

    target = TargetSpec(output_target="Wh", shape=(n_k, n_bands), k_points=n_k, n_bands=n_bands)
    spec = DataRepresentationSpec(
        input_repr="rho_grid",
        output_target=target.output_target,
        grid_shape=grid_shape,
        k_points=target.k_points,
        n_bands=target.n_bands,
    )
    assert validate_representation_compatibility(schema, spec) == []


def test_hdf5_field_schema_inspection_gracefully_skips_without_h5py(tmp_path):
    h5py = pytest.importorskip("h5py")
    grid_shape = (8, 8)
    n_samples = 2
    n_k = 5
    n_bands = 3
    path = tmp_path / "field.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("rho", data=np.zeros((n_samples, len(wedge_indices(grid_shape))), dtype=np.float32))
        handle.create_dataset("epsilon", data=np.ones((n_samples, *grid_shape), dtype=np.float32))
        handle.create_dataset("k_points", data=np.zeros((n_samples, n_k, 2), dtype=np.float32))
        handle.create_dataset("freqs", data=np.ones((n_samples, n_k, n_bands), dtype=np.float32))
        handle.create_dataset("efield", data=np.zeros((n_samples, n_bands, *grid_shape), dtype=np.float32))
        handle.create_dataset("hfield", data=np.zeros((n_samples, n_bands, *grid_shape), dtype=np.float32))
        handle.attrs["metadata_json"] = '{"source": "unit-test", "polarization": "TE"}'

    schema = infer_dataset_schema(path)
    assert schema["kind"] == "hdf5_field_or_ibz"
    assert schema["n_samples"] == n_samples
    assert schema["grid_shape"] == grid_shape
    assert schema["k_points"] == n_k
    assert schema["n_bands"] == n_bands

    field_schema = inspect_field_dataset(path)
    assert field_schema.has_vector_fields
    assert field_schema.n_samples == n_samples
    assert field_schema.grid_shape == grid_shape
    assert field_schema.metadata["source"] == "unit-test"
    assert field_schema.polarization == "TE"
    assert field_schema.missing_core_datasets == ()
    assert set(field_schema.missing_operator_context) == {
        "boundary_condition",
        "grid_spacing",
        "domain_metadata",
    }
    feasibility = field_schema.feasibility()
    assert feasibility["fno_field_unet_field"].ready_for_schema_smoke
    assert not feasibility["neurolight_adapted_field"].ready_for_full_claim
    assert "boundary_condition" in feasibility["maxwellnet_residual"].missing


def test_repository_field_tiny_schema_when_available():
    pytest.importorskip("h5py")
    path = Path("data/field_tiny_mpb.h5")
    if not path.exists():
        pytest.skip("data/field_tiny_mpb.h5 is not present in this checkout")

    field_schema = inspect_field_dataset(path)
    for name in CORE_FIELD_DATASETS:
        assert field_schema.has_field(name)
    assert field_schema.has_vector_fields
    assert field_schema.polarization in {"TE", "TM", "TE,TM"}
    assert field_schema.missing_operator_context
    assert set(OPERATOR_CONTEXT_FIELDS) >= set(field_schema.missing_operator_context)


def test_representation_compatibility_validation_and_transforms():
    grid_shape = (8, 8)
    n_wedge = len(wedge_indices(grid_shape))
    n_k = 4
    n_bands = 2
    schema = {
        "fields": {
            "rho": {"shape": (1, n_wedge), "dtype": "float32"},
            "Wh": {"shape": (1, n_k, n_bands), "dtype": "float32"},
        }
    }
    spec = DataRepresentationSpec(
        input_repr="image_epsilon",
        output_target="Wh",
        grid_shape=grid_shape,
        k_points=n_k,
        n_bands=n_bands,
    )
    assert validate_representation_compatibility(schema, spec) == []

    bad_spec = DataRepresentationSpec(input_repr="field_conditioned", output_target="near_field_te")
    assert validate_representation_compatibility(schema, bad_spec)

    sample = {
        "rho": np.eye(1, n_wedge, dtype=np.float32).reshape(n_wedge),
        "Wh": np.ones((n_k, n_bands), dtype=np.float32),
        "gap_width": 0.1,
    }
    assert make_input(sample, spec).shape == (*grid_shape, 1)
    assert make_target(sample, spec).shape == (n_k, n_bands)
    gaps = make_target(sample, DataRepresentationSpec(input_repr="rho_wedge", output_target="gap_descriptors"))
    assert gaps["gap_width"] == 0.1


def test_lift_torch_import_is_lazy(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    module = importlib.import_module("phc2d.data.lift_torch")
    assert module.__name__ == "phc2d.data.lift_torch"
    assert "torch" not in sys.modules
