"""Bloch-eigenfield record normalization and HDF5 serialization.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def hash_array(array: np.ndarray) -> str:
    """Return a shape- and dtype-aware content hash for record lineage."""

    arr = np.ascontiguousarray(np.asarray(array))
    header = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(header + b"\0" + arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class FieldEigenRecord:
    """One normalized `(structure, k-point, band)` TE/TM eigenfield record."""

    record_id: str
    source_sample_id: str
    source_sample_hash: str
    source_split: str
    rho: np.ndarray
    epsilon: np.ndarray
    k_point: np.ndarray
    k_path_id: str
    band_index_one_based: int
    eigenfrequency: float
    efield_complex: np.ndarray
    hfield_complex: np.ndarray
    polarization: str
    bloch_phase: np.ndarray
    lattice_constant: float
    domain_extent: np.ndarray
    dx: float
    dy: float
    field_normalization: dict[str, Any]
    solver_provenance: dict[str, Any]


def canonicalize_eigenfields(
    efield: np.ndarray,
    hfield: np.ndarray,
    *,
    epsilon: np.ndarray,
    domain_extent: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one epsilon-weighted E scale and global phase to both fields.

    Eigenmodes are arbitrary up to a complex scalar.  Applying the same scalar
    to E and H preserves their Maxwell relation, unlike normalizing the two
    fields independently.
    """

    e = np.asarray(efield, dtype=np.complex128)
    h = np.asarray(hfield, dtype=np.complex128)
    if e.ndim < 2 or h.ndim < 2:
        raise ValueError("MPB eigenfields must expose at least two spatial axes.")
    if tuple(e.shape[:2]) != tuple(h.shape[:2]):
        raise ValueError(f"E/H spatial shapes disagree: {e.shape} versus {h.shape}")
    extent = np.asarray(domain_extent, dtype=np.float64).reshape(-1)
    if extent.shape[0] != 2 or np.any(extent <= 0.0):
        raise ValueError(f"domain_extent must contain two positive values, got {extent.tolist()}")
    nx, ny = int(e.shape[0]), int(e.shape[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("MPB eigenfield spatial axes must be non-empty.")
    dx, dy = float(extent[0] / nx), float(extent[1] / ny)
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.ndim != 2 or np.any(~np.isfinite(eps)) or np.any(eps <= 0.0):
        raise ValueError("epsilon must be a finite positive two-dimensional material grid.")
    if nx % eps.shape[0] or ny % eps.shape[1]:
        raise ValueError(f"Cannot derive field epsilon grid {(nx, ny)} from material grid {eps.shape}: non-integer ratio.")
    eps_field = np.repeat(np.repeat(eps, nx // eps.shape[0], axis=0), ny // eps.shape[1], axis=1)
    while eps_field.ndim < e.ndim:
        eps_field = eps_field[..., None]
    l2 = float(np.sqrt(np.sum(eps_field * np.abs(e) ** 2) * dx * dy))
    if not np.isfinite(l2) or l2 <= 0.0:
        raise ValueError("Cannot normalize an all-zero or non-finite E eigenfield.")
    e_scaled = e / l2
    h_scaled = h / l2
    flat = e_scaled.reshape(-1)
    anchor_index = int(np.argmax(np.abs(flat)))
    anchor_before = complex(flat[anchor_index])
    phase = np.exp(-1j * np.angle(anchor_before))
    e_normalized = e_scaled * phase
    h_normalized = h_scaled * phase
    metadata = {
        "field_l2": "cell_volume_epsilon_weighted_l2_equals_one_using_efield",
        "electric_inner_product": "integral_conjugate_e1_dot_epsilon_e2_dv",
        "global_phase": "largest_magnitude_efield_component_real_and_nonnegative",
        "reference_field": "efield",
        "raw_efield_epsilon_weighted_l2": l2,
        "anchor_flat_index": anchor_index,
        "anchor_phase_rad_before": float(np.angle(anchor_before)),
        "anchor_value_after": [float(np.real(e_normalized.reshape(-1)[anchor_index])), float(np.imag(e_normalized.reshape(-1)[anchor_index]))],
        "dx": dx,
        "dy": dy,
    }
    return e_normalized, h_normalized, metadata


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("h5py is required to write Bloch eigenfield datasets.") from exc
    return h5py


def write_field_eigen_hdf5(
    path: str | Path,
    records: Sequence[FieldEigenRecord],
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Write schema-tagged native-complex field arrays for one or more records."""

    if not records:
        raise ValueError("At least one FieldEigenRecord is required.")
    h5py = _require_h5py()
    rows = list(records)
    e_shape = tuple(rows[0].efield_complex.shape)
    h_shape = tuple(rows[0].hfield_complex.shape)
    epsilon_shape = tuple(rows[0].epsilon.shape)
    rho_shape = tuple(rows[0].rho.shape)
    for row in rows:
        if tuple(row.efield_complex.shape) != e_shape or tuple(row.hfield_complex.shape) != h_shape:
            raise ValueError("All serialized field records must have a common E/H schema.")
        if tuple(row.epsilon.shape) != epsilon_shape or tuple(row.rho.shape) != rho_shape:
            raise ValueError("All serialized records must share rho and epsilon shapes.")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out, "w") as handle:
        handle.create_dataset("record_id", data=np.asarray([row.record_id for row in rows], dtype=object), dtype=string_dtype)
        handle.create_dataset("source_sample_id", data=np.asarray([row.source_sample_id for row in rows], dtype=object), dtype=string_dtype)
        handle.create_dataset("source_sample_hash", data=np.asarray([row.source_sample_hash for row in rows], dtype=object), dtype=string_dtype)
        handle.create_dataset("source_split", data=np.asarray([row.source_split for row in rows], dtype=object), dtype=string_dtype)
        handle.create_dataset("rho", data=np.stack([np.asarray(row.rho, dtype=np.float32) for row in rows], axis=0))
        handle.create_dataset("epsilon", data=np.stack([np.asarray(row.epsilon, dtype=np.float32) for row in rows], axis=0))
        handle.create_dataset("k_points", data=np.stack([np.asarray(row.k_point, dtype=np.float64) for row in rows], axis=0))
        handle.create_dataset("k_path_id", data=np.asarray([row.k_path_id for row in rows], dtype=object), dtype=string_dtype)
        handle.create_dataset("band_index", data=np.asarray([row.band_index_one_based for row in rows], dtype=np.int32))
        handle.create_dataset("eigenfrequency", data=np.asarray([row.eigenfrequency for row in rows], dtype=np.float64))
        efield_data = np.stack([row.efield_complex for row in rows], axis=0).astype(np.complex64)
        hfield_data = np.stack([row.hfield_complex for row in rows], axis=0).astype(np.complex64)
        # Keep explicit names rather than HDF5 hard links: generic schema
        # inspection visits a shared object only once, which would otherwise
        # hide either the compatibility name or the explicit-complex name.
        handle.create_dataset("efield_complex", data=efield_data)
        handle.create_dataset("hfield_complex", data=hfield_data)
        handle.create_dataset("efield", data=efield_data)
        handle.create_dataset("hfield", data=hfield_data)
        handle.create_dataset("polarization", data=np.asarray([row.polarization for row in rows], dtype=object), dtype=string_dtype)
        handle.create_dataset("bloch_phase", data=np.stack([np.asarray(row.bloch_phase, dtype=np.float64) for row in rows], axis=0))
        handle.create_dataset("lattice_constant", data=np.asarray([row.lattice_constant for row in rows], dtype=np.float64))
        handle.create_dataset("domain_extent", data=np.stack([np.asarray(row.domain_extent, dtype=np.float64) for row in rows], axis=0))
        handle.create_dataset("dx", data=np.asarray([row.dx for row in rows], dtype=np.float64))
        handle.create_dataset("dy", data=np.asarray([row.dy for row in rows], dtype=np.float64))
        handle.create_dataset(
            "field_normalization_json",
            data=np.asarray([json.dumps(_jsonable(row.field_normalization), ensure_ascii=False, sort_keys=True) for row in rows], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "solver_provenance_json",
            data=np.asarray([json.dumps(_jsonable(row.solver_provenance), ensure_ascii=False, sort_keys=True) for row in rows], dtype=object),
            dtype=string_dtype,
        )
        handle.attrs["metadata_json"] = json.dumps(_jsonable(dict(metadata)), ensure_ascii=False, sort_keys=True)
        handle.attrs["complex_storage"] = "native_hdf5_complex; efield/hfield duplicate *_complex for compatibility"
        handle.attrs["boundary_condition"] = "bloch_periodic"
        handle.attrs["field_schema_version"] = "phc2d.field_eigen.v2"
    return out


__all__ = [
    "FieldEigenRecord",
    "canonicalize_eigenfields",
    "hash_array",
    "write_field_eigen_hdf5",
]
