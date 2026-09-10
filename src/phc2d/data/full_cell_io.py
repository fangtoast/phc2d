"""HDF5 readers and writers for full-cell datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from phc2d.data.full_cell_hashing import (
    canonicalize_structure_payload,
    hash_raster_content,
)
from phc2d.data.full_cell_raster import (
    DIRECT_CELL_CENTER_V1,
    MASTER_BLOCK_MEAN_V1,
    build_full_cell_raster_pyramid,
)
from phc2d.data.full_cell_spec import FULL_CELL_SCHEMA_VERSION, FullCellStructureSpec


_canonical_json = canonicalize_structure_payload
FULL_CELL_DATASET_SCHEMA_V2 = "phc2d.full_cell_dataset.v2"


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
    material_area_fraction_grids: dict[int, np.ndarray] = field(default_factory=dict)
    material_binary_grids: dict[int, np.ndarray] = field(default_factory=dict)
    content_hashes: dict[int, tuple[str, ...]] = field(default_factory=dict)
    schema_version: str = FULL_CELL_SCHEMA_VERSION


def _rasterize_structure_resolutions(
    task: tuple[
        int,
        FullCellStructureSpec,
        tuple[int, ...],
        str,
        int,
        int,
    ],
) -> tuple[int, dict[str, np.ndarray], dict[int, str]]:
    """Build named raster fields and content hashes for one structure."""

    index, structure, resolutions, method, source_resolution, quantization_levels = task
    fields: dict[str, np.ndarray] = {}
    hashes: dict[int, str] = {}
    if method == DIRECT_CELL_CENTER_V1:
        for resolution in resolutions:
            weight = structure.rasterize((resolution, resolution)).astype(
                np.float32,
                copy=False,
            )
            epsilon = (
                float(structure.epsilon_air)
                + (float(structure.epsilon_material) - float(structure.epsilon_air))
                * weight
            ).astype(np.float32, copy=False)
            fields[f"material_weight_grid_{resolution}"] = weight
            fields[f"epsilon_grid_{resolution}"] = epsilon
        return index, fields, hashes
    if method != MASTER_BLOCK_MEAN_V1:
        raise ValueError(f"unsupported raster method: {method}")
    pyramid = build_full_cell_raster_pyramid(
        structure,
        resolutions=resolutions,
        source_resolution=source_resolution,
        method=method,
    )
    for resolution, grid in pyramid.grids.items():
        if structure.material_kind == "binary" and resolution == source_resolution:
            normalized = np.asarray(grid >= 0.5, dtype=np.uint8)
            fields[f"material_binary_grid_{resolution}"] = normalized
            hash_kind = "binary"
            hash_grid = normalized
        elif structure.material_kind == "binary":
            normalized = np.asarray(grid, dtype=np.float32)
            fields[f"material_area_fraction_grid_{resolution}"] = normalized
            hash_kind = "area_fraction"
            hash_grid = normalized
        else:
            normalized = np.asarray(grid, dtype=np.float32)
            fields[f"material_weight_grid_{resolution}"] = normalized
            hash_kind = "continuous"
            hash_grid = normalized
        hashes[resolution] = hash_raster_content(
            hash_grid,
            material_kind=hash_kind,
            quantization_levels=quantization_levels,
        )
    return index, fields, hashes


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
    raster_method: str = DIRECT_CELL_CENTER_V1,
    raster_source_resolution: int = 256,
    content_hash_quantization_levels: int = 65536,
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
    resolutions = tuple(sorted({int(value) for value in raster_resolutions}))
    if not resolutions or min(resolutions) <= 0:
        raise ValueError("raster resolutions must be positive")
    method = str(raster_method)
    raster_rows = [
        _rasterize_structure_resolutions(
            (
                index,
                item,
                resolutions,
                method,
                int(raster_source_resolution),
                int(content_hash_quantization_levels),
            )
        )
        for index, item in enumerate(items)
    ]
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = (
            FULL_CELL_SCHEMA_VERSION
            if method == DIRECT_CELL_CENTER_V1
            else FULL_CELL_DATASET_SCHEMA_V2
        )
        handle.attrs["metadata_json"] = _canonical_json(metadata)
        handle.attrs["material_parameterization"] = "analytic_periodic_full_cell"
        handle.attrs["raster_method"] = method
        handle.attrs["raster_source_resolution"] = int(raster_source_resolution)
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
        if method == MASTER_BLOCK_MEAN_V1:
            handle.create_dataset(
                "parameter_hash",
                data=np.asarray([item.parameter_hash for item in items], dtype=object),
                dtype=string_dtype,
            )
            handle.create_dataset(
                "geometry_hash",
                data=np.asarray([item.geometry_hash for item in items], dtype=object),
                dtype=string_dtype,
            )
        field_names = sorted({name for _, fields, _ in raster_rows for name in fields})
        for name in field_names:
            grids = np.stack([fields[name] for _, fields, _ in raster_rows], axis=0)
            resolution = int(name.rsplit("_", 1)[1])
            handle.create_dataset(
                name,
                data=grids,
                compression="gzip",
                shuffle=True,
                chunks=(1, resolution, resolution),
            )
        if method == MASTER_BLOCK_MEAN_V1:
            for resolution in resolutions:
                handle.create_dataset(
                    f"content_hash_r{resolution}",
                    data=np.asarray(
                        [hashes[resolution] for _, _, hashes in raster_rows],
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )
    return out


def write_full_cell_hdf5_streaming(
    path: str | Path,
    *,
    structures: Sequence[FullCellStructureSpec],
    band_shape: Sequence[int],
    band_loader: Callable[[int], np.ndarray],
    k_points: np.ndarray,
    k_grid_index: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    raster_resolutions: Sequence[int],
    metadata: Mapping[str, Any],
    raster_workers: int = 1,
    row_chunk_size: int = 256,
    progress_fn: Callable[[str, int, int], None] | None = None,
    raster_method: str = DIRECT_CELL_CENTER_V1,
    raster_source_resolution: int = 256,
    content_hash_quantization_levels: int = 65536,
) -> Path:
    """Atomically build a full-cell HDF5 with bounded memory and parallel rasters."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required to write full-cell datasets") from exc

    items = tuple(structures)
    n_items = len(items)
    shape = tuple(int(value) for value in band_shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("band_shape must be [n_k_points, n_bands]")
    points = np.asarray(k_points, dtype=np.float64)
    grid_index = np.asarray(k_grid_index, dtype=np.int64)
    if points.shape != (shape[0], 2) or grid_index.shape != points.shape:
        raise ValueError("k_points and k_grid_index do not match band_shape")
    split_arrays = tuple(
        np.asarray(value, dtype=np.int64).reshape(-1) for value in (idx_train, idx_val, idx_test)
    )
    merged = np.concatenate(split_arrays) if split_arrays else np.empty(0, dtype=np.int64)
    if sorted(int(value) for value in merged) != list(range(n_items)):
        raise ValueError("train/val/test indices must partition all structures exactly once")
    resolutions = tuple(sorted({int(value) for value in raster_resolutions}))
    if not resolutions or min(resolutions) <= 0:
        raise ValueError("raster resolutions must be positive")
    method = str(raster_method)
    source_resolution = int(raster_source_resolution)
    quantization_levels = int(content_hash_quantization_levels)
    if method == MASTER_BLOCK_MEAN_V1:
        if source_resolution not in resolutions:
            raise ValueError("master block-mean rasters must include source_resolution")
        if len({item.material_kind for item in items}) > 1:
            raise ValueError("one v2 HDF5 lane cannot mix binary and continuous material kinds")
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else int(os.cpu_count() or 1)
    )
    workers = max(1, min(int(raster_workers), n_items or 1, available_cpus))
    chunk_size = max(1, int(row_chunk_size))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    incomplete = out.with_name(f".{out.name}.incomplete")
    if incomplete.exists():
        raise RuntimeError(f"Incomplete finalizer output already exists: {incomplete}")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    max_params = max((len(item.parameter_vector) for item in items), default=0)

    with h5py.File(incomplete, "w") as handle:
        handle.attrs["schema_version"] = (
            FULL_CELL_SCHEMA_VERSION
            if method == DIRECT_CELL_CENTER_V1
            else FULL_CELL_DATASET_SCHEMA_V2
        )
        handle.attrs["metadata_json"] = _canonical_json(metadata)
        handle.attrs["material_parameterization"] = "analytic_periodic_full_cell"
        handle.attrs["raster_method"] = method
        handle.attrs["raster_source_resolution"] = source_resolution
        sample_ds = handle.create_dataset("sample_id", shape=(n_items,), dtype=string_dtype)
        family_ds = handle.create_dataset("family_id", shape=(n_items,), dtype=string_dtype)
        kind_ds = handle.create_dataset("material_kind", shape=(n_items,), dtype=string_dtype)
        structure_ds = handle.create_dataset("structure_json", shape=(n_items,), dtype=string_dtype)
        parameter_hash_ds = (
            handle.create_dataset("parameter_hash", shape=(n_items,), dtype=string_dtype)
            if method == MASTER_BLOCK_MEAN_V1
            else None
        )
        geometry_hash_ds = (
            handle.create_dataset("geometry_hash", shape=(n_items,), dtype=string_dtype)
            if method == MASTER_BLOCK_MEAN_V1
            else None
        )
        params_ds = handle.create_dataset(
            "params", shape=(n_items, max_params), dtype=np.float32, compression="gzip", shuffle=True
        )
        mask_ds = handle.create_dataset(
            "param_mask", shape=(n_items, max_params), dtype=np.uint8, compression="gzip", shuffle=True
        )
        bands_ds = handle.create_dataset(
            "Wh",
            shape=(n_items, *shape),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            chunks=(1, *shape),
        )
        handle.create_dataset("k_points", data=points)
        handle.create_dataset("k_grid_index", data=grid_index)
        handle.create_dataset("idx_train", data=split_arrays[0])
        handle.create_dataset("idx_val", data=split_arrays[1])
        handle.create_dataset("idx_test", data=split_arrays[2])
        raster_specs: dict[str, tuple[int, Any]] = {}
        if method == DIRECT_CELL_CENTER_V1:
            for resolution in resolutions:
                raster_specs[f"material_weight_grid_{resolution}"] = (resolution, np.float32)
                raster_specs[f"epsilon_grid_{resolution}"] = (resolution, np.float32)
        elif method == MASTER_BLOCK_MEAN_V1:
            material_kind = items[0].material_kind if items else "continuous"
            for resolution in resolutions:
                if material_kind == "binary" and resolution == source_resolution:
                    raster_specs[f"material_binary_grid_{resolution}"] = (resolution, np.uint8)
                elif material_kind == "binary":
                    raster_specs[f"material_area_fraction_grid_{resolution}"] = (resolution, np.float32)
                else:
                    raster_specs[f"material_weight_grid_{resolution}"] = (resolution, np.float32)
        else:
            raise ValueError(f"unsupported raster method: {method}")
        raster_ds = {
            name: handle.create_dataset(
                name,
                shape=(n_items, resolution, resolution),
                dtype=dtype,
                compression="gzip",
                shuffle=True,
                chunks=(1, resolution, resolution),
            )
            for name, (resolution, dtype) in raster_specs.items()
        }
        content_hash_ds = {
            resolution: handle.create_dataset(
                f"content_hash_r{resolution}",
                shape=(n_items,),
                dtype=string_dtype,
            )
            for resolution in resolutions
        } if method == MASTER_BLOCK_MEAN_V1 else {}

        for start in range(0, n_items, chunk_size):
            end = min(n_items, start + chunk_size)
            rows = items[start:end]
            sample_ds[start:end] = [item.sample_id for item in rows]
            family_ds[start:end] = [item.generator_family for item in rows]
            kind_ds[start:end] = [item.material_kind for item in rows]
            structure_ds[start:end] = [_canonical_json(item.to_mapping()) for item in rows]
            if parameter_hash_ds is not None and geometry_hash_ds is not None:
                parameter_hash_ds[start:end] = [item.parameter_hash for item in rows]
                geometry_hash_ds[start:end] = [item.geometry_hash for item in rows]
            params = np.zeros((end - start, max_params), dtype=np.float32)
            mask = np.zeros((end - start, max_params), dtype=np.uint8)
            band_rows = np.empty((end - start, *shape), dtype=np.float32)
            for offset, item in enumerate(rows):
                width = len(item.parameter_vector)
                params[offset, :width] = np.asarray(item.parameter_vector, dtype=np.float32)
                mask[offset, :width] = np.asarray(item.parameter_mask, dtype=np.uint8)
                band_rows[offset] = np.asarray(band_loader(start + offset), dtype=np.float32)
            if not np.all(np.isfinite(band_rows)):
                raise RuntimeError("Full-cell bands contain NaN or Inf")
            if np.any(np.diff(band_rows, axis=-1) < -1.0e-7):
                raise RuntimeError("Full-cell bands violate ascending band order")
            params_ds[start:end] = params
            mask_ds[start:end] = mask
            bands_ds[start:end] = band_rows
            handle.flush()
            if progress_fn is not None:
                progress_fn("bands", end, n_items)

        tasks = iter(
            (
                index,
                item,
                resolutions,
                method,
                source_resolution,
                quantization_levels,
            )
            for index, item in enumerate(items)
        )
        if workers == 1:
            generated = map(_rasterize_structure_resolutions, tasks)
            for completed, (index, fields, hashes) in enumerate(generated, start=1):
                for name, grid in fields.items():
                    raster_ds[name][index] = grid
                for resolution, digest in hashes.items():
                    content_hash_ds[resolution][index] = digest
                if completed % chunk_size == 0:
                    handle.flush()
                    if progress_fn is not None:
                        progress_fn("rasters", completed, n_items)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            ) as executor:
                futures: dict[Any, None] = {}
                max_in_flight = max(1, workers * 2)

                def fill() -> None:
                    while len(futures) < max_in_flight:
                        try:
                            task = next(tasks)
                        except StopIteration:
                            break
                        future = executor.submit(_rasterize_structure_resolutions, task)
                        futures[future] = None

                completed = 0
                fill()
                while futures:
                    done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        index, fields, hashes = future.result()
                        for name, grid in fields.items():
                            raster_ds[name][index] = grid
                        for resolution, digest in hashes.items():
                            content_hash_ds[resolution][index] = digest
                        completed += 1
                        if completed % chunk_size == 0:
                            handle.flush()
                            if progress_fn is not None:
                                progress_fn("rasters", completed, n_items)
                    fill()
        if progress_fn is not None:
            progress_fn("rasters", n_items, n_items)
        handle.flush()
    os.replace(incomplete, out)
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
        schema_version = decode(handle.attrs.get("schema_version", ""))
        if schema_version not in {FULL_CELL_SCHEMA_VERSION, FULL_CELL_DATASET_SCHEMA_V2}:
            raise ValueError("HDF5 file is not a supported phc2d full-cell dataset")
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
        area_fraction = {
            int(name.rsplit("_", 1)[1]): np.asarray(handle[name], dtype=np.float32)
            for name in handle
            if name.startswith("material_area_fraction_grid_")
        }
        binary = {
            int(name.rsplit("_", 1)[1]): np.asarray(handle[name], dtype=np.uint8)
            for name in handle
            if name.startswith("material_binary_grid_")
        }
        content_hashes = {
            int(name.removeprefix("content_hash_r")): tuple(
                decode(value) for value in handle[name][:]
            )
            for name in handle
            if name.startswith("content_hash_r")
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
            material_area_fraction_grids=area_fraction,
            material_binary_grids=binary,
            content_hashes=content_hashes,
            schema_version=schema_version,
        )


__all__ = [
    "FULL_CELL_DATASET_SCHEMA_V2",
    "FullCellDatasetBundle",
    "load_full_cell_hdf5",
    "write_full_cell_hdf5",
    "write_full_cell_hdf5_streaming",
]
