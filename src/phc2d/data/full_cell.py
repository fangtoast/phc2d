"""Full-cell structure and HDF5 contracts without p4m assumptions.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FULL_CELL_SCHEMA_VERSION = "phc2d.full_cell.v1"
SUPPORTED_FULL_CELL_FAMILIES = frozenset(
    {
        "graded_rbf_fourier",
        "binary_polar_holes",
        "binary_spectral_threshold",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_full_bz_grid(n_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a centered, duplicate-free square-BZ grid in reduced coordinates."""

    n = int(n_side)
    if n <= 0 or n % 2 == 0:
        raise ValueError("full BZ n_side must be a positive odd integer")
    half = n // 2
    indices = np.asarray([(i, j) for i in range(-half, half + 1) for j in range(-half, half + 1)], dtype=np.int64)
    points = indices.astype(np.float64) / float(n)
    return points, indices


def _grid_coordinates(grid_shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain two dimensions")
    rows, cols = (int(grid_shape[0]), int(grid_shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape dimensions must be positive")
    x = -0.5 + (np.arange(cols, dtype=np.float64) + 0.5) / float(cols)
    y = -0.5 + (np.arange(rows, dtype=np.float64) + 0.5) / float(rows)
    return np.meshgrid(x, y, indexing="xy")


def _fourier_field(x: np.ndarray, y: np.ndarray, modes: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=np.float64)
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    for row in modes:
        phase = 2.0 * np.pi * (int(row["mx"]) * x_arr + int(row["my"]) * y_arr)
        result += float(row.get("cos", 0.0)) * np.cos(phase)
        result += float(row.get("sin", 0.0)) * np.sin(phase)
    return result


def _wrapped_delta(value: np.ndarray, center: float) -> np.ndarray:
    return np.mod(np.asarray(value, dtype=np.float64) - float(center) + 0.5, 1.0) - 0.5


def _graded_weight(x: np.ndarray, y: np.ndarray, payload: Mapping[str, Any]) -> np.ndarray:
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    field = np.zeros_like(x_arr, dtype=np.float64)
    for blob in payload.get("blobs", []):
        dx = _wrapped_delta(x_arr, float(blob["center_x"]))
        dy = _wrapped_delta(y_arr, float(blob["center_y"]))
        angle = float(blob["angle"])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xr = cos_a * dx + sin_a * dy
        yr = -sin_a * dx + cos_a * dy
        sigma_x = max(float(blob["sigma_x"]), 1.0e-8)
        sigma_y = max(float(blob["sigma_y"]), 1.0e-8)
        field += float(blob["amplitude"]) * np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    field += float(payload.get("fourier_strength", 0.0)) * _fourier_field(x_arr, y_arr, payload.get("modes", []))
    temperature = max(float(payload["temperature"]), 1.0e-8)
    logits = np.clip((field - float(payload["bias"])) / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _polar_radius(theta: np.ndarray, hole: Mapping[str, Any]) -> np.ndarray:
    perturbation = np.zeros_like(theta, dtype=np.float64)
    for row in hole.get("modes", []):
        order = int(row["order"])
        perturbation += float(row.get("cos", 0.0)) * np.cos(order * theta)
        perturbation += float(row.get("sin", 0.0)) * np.sin(order * theta)
    return float(hole["radius"]) * np.exp(perturbation)


def _binary_polar_weight(x: np.ndarray, y: np.ndarray, payload: Mapping[str, Any]) -> np.ndarray:
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    air = np.zeros_like(x_arr, dtype=bool)
    for hole in payload.get("holes", []):
        dx = _wrapped_delta(x_arr, float(hole["center_x"]))
        dy = _wrapped_delta(y_arr, float(hole["center_y"]))
        theta = np.arctan2(dy, dx)
        radius = np.sqrt(dx * dx + dy * dy)
        air |= radius <= _polar_radius(theta, hole)
    return (~air).astype(np.float64)


def _binary_spectral_weight(x: np.ndarray, y: np.ndarray, payload: Mapping[str, Any]) -> np.ndarray:
    field = _fourier_field(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), payload.get("modes", []))
    return (field >= float(payload["threshold"])).astype(np.float64)


@dataclass(frozen=True)
class FullCellStructureSpec:
    """One periodic full-cell material distribution reconstructed from parameters."""

    sample_id: str
    generator_family: str
    material_kind: str
    parameters: dict[str, Any]
    parameter_names: tuple[str, ...]
    parameter_vector: tuple[float, ...]
    parameter_mask: tuple[bool, ...]
    epsilon_air: float = 1.0
    epsilon_material: float = 8.9
    schema_version: str = FULL_CELL_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generator_family not in SUPPORTED_FULL_CELL_FAMILIES:
            raise ValueError(f"Unsupported full-cell generator family: {self.generator_family}")
        if self.material_kind not in {"continuous", "binary"}:
            raise ValueError("material_kind must be continuous or binary")
        if len(self.parameter_names) != len(self.parameter_vector) or len(self.parameter_mask) != len(self.parameter_vector):
            raise ValueError("parameter names, values, and mask must have equal length")
        if not float(self.epsilon_material) > float(self.epsilon_air) > 0.0:
            raise ValueError("epsilon_material must exceed positive epsilon_air")

    @property
    def geometry_hash(self) -> str:
        payload = self.to_mapping()
        payload.pop("metadata", None)
        return _sha256_text(_canonical_json(payload))

    def evaluate_material_weight(self, x: Any, y: Any) -> np.ndarray:
        if self.generator_family == "graded_rbf_fourier":
            weight = _graded_weight(np.asarray(x), np.asarray(y), self.parameters)
        elif self.generator_family == "binary_polar_holes":
            weight = _binary_polar_weight(np.asarray(x), np.asarray(y), self.parameters)
        else:
            weight = _binary_spectral_weight(np.asarray(x), np.asarray(y), self.parameters)
        return np.clip(np.asarray(weight, dtype=np.float64), 0.0, 1.0)

    def rasterize(self, grid_shape: Sequence[int]) -> np.ndarray:
        x, y = _grid_coordinates(grid_shape)
        return self.evaluate_material_weight(x, y).astype(np.float32)

    def epsilon_grid(self, grid_shape: Sequence[int]) -> np.ndarray:
        weight = self.rasterize(grid_shape)
        return (float(self.epsilon_air) + (float(self.epsilon_material) - float(self.epsilon_air)) * weight).astype(np.float32)

    def epsilon_at(self, x: float, y: float) -> float:
        weight = float(np.asarray(self.evaluate_material_weight(float(x), float(y))).reshape(()))
        return float(self.epsilon_air + (self.epsilon_material - self.epsilon_air) * weight)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "generator_family": self.generator_family,
            "material_kind": self.material_kind,
            "parameters": _jsonable(self.parameters),
            "parameter_names": list(self.parameter_names),
            "parameter_vector": list(self.parameter_vector),
            "parameter_mask": list(self.parameter_mask),
            "epsilon_air": float(self.epsilon_air),
            "epsilon_material": float(self.epsilon_material),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FullCellStructureSpec":
        return cls(
            schema_version=str(value.get("schema_version", FULL_CELL_SCHEMA_VERSION)),
            sample_id=str(value["sample_id"]),
            generator_family=str(value["generator_family"]),
            material_kind=str(value["material_kind"]),
            parameters=dict(value.get("parameters") or {}),
            parameter_names=tuple(str(item) for item in value.get("parameter_names", ())),
            parameter_vector=tuple(float(item) for item in value.get("parameter_vector", ())),
            parameter_mask=tuple(bool(item) for item in value.get("parameter_mask", ())),
            epsilon_air=float(value.get("epsilon_air", 1.0)),
            epsilon_material=float(value.get("epsilon_material", 8.9)),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class FullCellDatasetBundle:
    """Loaded full-cell band dataset with explicit physical coordinates."""

    structures: tuple[FullCellStructureSpec, ...]
    Wh: np.ndarray
    k_points: np.ndarray
    k_grid_index: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    material_weight_grids: dict[int, np.ndarray]
    epsilon_grids: dict[int, np.ndarray]
    metadata: dict[str, Any]


def write_full_cell_hdf5(
    path: str | Path,
    *,
    structures: Sequence[FullCellStructureSpec],
    Wh: np.ndarray,
    k_points: np.ndarray,
    k_grid_index: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    raster_resolutions: Sequence[int],
    metadata: Mapping[str, Any],
) -> Path:
    """Write one provenance-complete full-cell band dataset."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required to write full-cell datasets") from exc

    items = tuple(structures)
    bands = np.asarray(Wh, dtype=np.float32)
    points = np.asarray(k_points, dtype=np.float64)
    grid_index = np.asarray(k_grid_index, dtype=np.int64)
    if bands.ndim != 3 or bands.shape[0] != len(items) or bands.shape[1] != points.shape[0]:
        raise ValueError("Wh must have shape [n_structures, n_k_points, n_bands]")
    if points.ndim != 2 or points.shape[1] != 2 or grid_index.shape != points.shape:
        raise ValueError("k_points and k_grid_index must have matching [n_k_points, 2] shapes")
    split_arrays = tuple(np.asarray(value, dtype=np.int64).reshape(-1) for value in (idx_train, idx_val, idx_test))
    merged = np.concatenate(split_arrays) if split_arrays else np.empty(0, dtype=np.int64)
    if sorted(int(value) for value in merged) != list(range(len(items))):
        raise ValueError("train/val/test indices must partition all structures exactly once")

    max_params = max((len(item.parameter_vector) for item in items), default=0)
    params = np.zeros((len(items), max_params), dtype=np.float32)
    mask = np.zeros((len(items), max_params), dtype=np.uint8)
    for index, item in enumerate(items):
        width = len(item.parameter_vector)
        params[index, :width] = np.asarray(item.parameter_vector, dtype=np.float32)
        mask[index, :width] = np.asarray(item.parameter_mask, dtype=np.uint8)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = FULL_CELL_SCHEMA_VERSION
        handle.attrs["metadata_json"] = _canonical_json(metadata)
        handle.attrs["material_parameterization"] = "analytic_periodic_full_cell"
        handle.create_dataset("sample_id", data=np.asarray([item.sample_id for item in items], dtype=object), dtype=string_dtype)
        handle.create_dataset("family_id", data=np.asarray([item.generator_family for item in items], dtype=object), dtype=string_dtype)
        handle.create_dataset("material_kind", data=np.asarray([item.material_kind for item in items], dtype=object), dtype=string_dtype)
        handle.create_dataset("structure_json", data=np.asarray([_canonical_json(item.to_mapping()) for item in items], dtype=object), dtype=string_dtype)
        handle.create_dataset("params", data=params, compression="gzip", shuffle=True)
        handle.create_dataset("param_mask", data=mask, compression="gzip", shuffle=True)
        handle.create_dataset("Wh", data=bands, compression="gzip", shuffle=True, chunks=(1, bands.shape[1], bands.shape[2]))
        handle.create_dataset("k_points", data=points)
        handle.create_dataset("k_grid_index", data=grid_index)
        handle.create_dataset("idx_train", data=split_arrays[0])
        handle.create_dataset("idx_val", data=split_arrays[1])
        handle.create_dataset("idx_test", data=split_arrays[2])
        for resolution in sorted({int(value) for value in raster_resolutions}):
            if resolution <= 0:
                raise ValueError("raster resolutions must be positive")
            weight = np.stack([item.rasterize((resolution, resolution)) for item in items], axis=0).astype(np.float32)
            epsilon = np.stack([item.epsilon_grid((resolution, resolution)) for item in items], axis=0).astype(np.float32)
            handle.create_dataset(f"material_weight_grid_{resolution}", data=weight, compression="gzip", shuffle=True, chunks=(1, resolution, resolution))
            handle.create_dataset(f"epsilon_grid_{resolution}", data=epsilon, compression="gzip", shuffle=True, chunks=(1, resolution, resolution))
    return out


def load_full_cell_hdf5(path: str | Path) -> FullCellDatasetBundle:
    """Load a full-cell dataset without importing h5py until requested."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required to load full-cell datasets") from exc

    def decode(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    with h5py.File(Path(path), "r") as handle:
        if str(handle.attrs.get("schema_version", "")) != FULL_CELL_SCHEMA_VERSION:
            raise ValueError("HDF5 file is not a phc2d full-cell v1 dataset")
        structures = tuple(FullCellStructureSpec.from_mapping(json.loads(decode(value))) for value in handle["structure_json"][:])
        weight = {
            int(name.rsplit("_", 1)[1]): np.asarray(handle[name], dtype=np.float32)
            for name in handle
            if name.startswith("material_weight_grid_")
        }
        epsilon = {
            int(name.rsplit("_", 1)[1]): np.asarray(handle[name], dtype=np.float32)
            for name in handle
            if name.startswith("epsilon_grid_")
        }
        metadata = json.loads(decode(handle.attrs.get("metadata_json", "{}")))
        return FullCellDatasetBundle(
            structures=structures,
            Wh=np.asarray(handle["Wh"], dtype=np.float32),
            k_points=np.asarray(handle["k_points"], dtype=np.float64),
            k_grid_index=np.asarray(handle["k_grid_index"], dtype=np.int64),
            idx_train=np.asarray(handle["idx_train"], dtype=np.int64),
            idx_val=np.asarray(handle["idx_val"], dtype=np.int64),
            idx_test=np.asarray(handle["idx_test"], dtype=np.int64),
            material_weight_grids=weight,
            epsilon_grids=epsilon,
            metadata=metadata,
        )


__all__ = [
    "FULL_CELL_SCHEMA_VERSION",
    "FullCellDatasetBundle",
    "FullCellStructureSpec",
    "SUPPORTED_FULL_CELL_FAMILIES",
    "build_full_bz_grid",
    "load_full_cell_hdf5",
    "write_full_cell_hdf5",
]
