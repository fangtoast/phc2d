"""Canonical serialization and versioned hashes for full-cell data.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


CONTENT_HASH_VERSION = "phc2d.full_cell.content_hash.v1"
PARAMETER_HASH_VERSION = "phc2d.full_cell.parameter_hash.v1"


def jsonable_full_cell_value(value: Any) -> Any:
    """Convert nested NumPy-bearing values into deterministic JSON values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): jsonable_full_cell_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable_full_cell_value(item) for item in value]
    return value


def canonicalize_structure_payload(value: Any) -> str:
    """Serialize a full-cell payload with stable keys and separators."""

    return json.dumps(
        jsonable_full_cell_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 hex digest of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_structure_parameters(value: Any) -> str:
    """Hash canonical analytic parameters without sample or split identity."""

    payload = {
        "algorithm": PARAMETER_HASH_VERSION,
        "parameters": jsonable_full_cell_value(value),
    }
    return sha256_text(canonicalize_structure_payload(payload))


def hash_structure_geometry(value: Mapping[str, Any]) -> str:
    """Preserve the legacy record hash while excluding mutable metadata."""

    payload = dict(value)
    payload.pop("metadata", None)
    return sha256_text(canonicalize_structure_payload(payload))


def _little_endian_bytes(array: np.ndarray) -> bytes:
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().view(dtype.newbyteorder("<"))
    return np.ascontiguousarray(array).tobytes(order="C")


def hash_raster_content(
    grid: np.ndarray,
    *,
    material_kind: str,
    quantization_levels: int = 65536,
) -> str:
    """Hash one raster with explicit shape, dtype, order, and quantization."""

    array = np.asarray(grid)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("content hashing expects a non-empty 2D raster")
    if not np.all(np.isfinite(array)):
        raise ValueError("content hashing rejects NaN or infinite values")
    kind = str(material_kind)
    if kind == "binary":
        normalized = np.asarray(array >= 0.5, dtype=np.uint8)
        encoded = np.packbits(normalized.reshape(-1), bitorder="little")
        encoding = "binary_uint8_packbits_little"
    elif kind in {"continuous", "area_fraction"}:
        levels = int(quantization_levels)
        if levels < 2 or levels > 65536:
            raise ValueError("quantization_levels must be in [2,65536]")
        clipped = np.clip(np.asarray(array, dtype=np.float64), 0.0, 1.0)
        encoded = np.rint(clipped * float(levels - 1)).astype("<u2")
        encoding = f"unit_interval_uint16_round_{levels}"
    else:
        raise ValueError(f"unsupported material_kind for content hash: {kind}")
    header = canonicalize_structure_payload(
        {
            "algorithm": CONTENT_HASH_VERSION,
            "shape": list(array.shape),
            "axis_order": "row_y_col_x",
            "encoding": encoding,
            "nan_policy": "reject",
            "byte_order": "little",
        }
    ).encode("utf-8")
    return hashlib.sha256(header + b"\n" + _little_endian_bytes(encoded)).hexdigest()


__all__ = [
    "CONTENT_HASH_VERSION",
    "PARAMETER_HASH_VERSION",
    "canonicalize_structure_payload",
    "hash_raster_content",
    "hash_structure_geometry",
    "hash_structure_parameters",
    "jsonable_full_cell_value",
    "sha256_text",
]
