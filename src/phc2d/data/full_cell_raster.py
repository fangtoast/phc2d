"""Versioned raster contracts for analytic full-cell structures.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from phc2d.data.full_cell_fourier import periodic_cell_centers


DIRECT_CELL_CENTER_V1 = "direct_cell_center_v1"
MASTER_BLOCK_MEAN_V1 = "master_block_mean_v1"


class MaterialWeightEvaluator(Protocol):
    """Minimal analytic structure interface required by rasterization."""

    material_kind: str

    def evaluate_material_weight(self, x: object, y: object) -> np.ndarray: ...


@dataclass(frozen=True)
class FullCellRasterPyramid:
    """One immutable set of grids derived by a named raster strategy."""

    source_resolution: int
    grids: Mapping[int, np.ndarray]
    method: str


def block_mean_downsample(master: np.ndarray, target_resolution: int) -> np.ndarray:
    """Area-average a square master grid directly to an integer divisor."""

    array = np.asarray(master)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("master grid must be square")
    source = int(array.shape[0])
    target = int(target_resolution)
    if target <= 0 or source % target != 0:
        raise ValueError("target_resolution must be a positive divisor of source")
    factor = source // target
    return array.reshape(target, factor, target, factor).mean(axis=(1, 3))


def rasterize_full_cell(
    structure: MaterialWeightEvaluator,
    grid_shape: Sequence[int],
) -> np.ndarray:
    """Directly sample an analytic structure at periodic cell centres."""

    x, y = periodic_cell_centers(grid_shape)
    return np.asarray(structure.evaluate_material_weight(x, y), dtype=np.float32)


def build_full_cell_raster_pyramid(
    structure: MaterialWeightEvaluator,
    *,
    resolutions: Sequence[int],
    source_resolution: int = 256,
    method: str = MASTER_BLOCK_MEAN_V1,
) -> FullCellRasterPyramid:
    """Build direct or anti-aliased multi-resolution material representations."""

    requested = tuple(sorted({int(value) for value in resolutions}))
    if not requested or min(requested) <= 0:
        raise ValueError("resolutions must contain positive integers")
    strategy = str(method)
    if strategy == DIRECT_CELL_CENTER_V1:
        grids = {
            resolution: rasterize_full_cell(structure, (resolution, resolution))
            for resolution in requested
        }
        return FullCellRasterPyramid(max(requested), grids, strategy)
    if strategy != MASTER_BLOCK_MEAN_V1:
        raise ValueError(f"unsupported raster method: {strategy}")
    source = int(source_resolution)
    if source <= 0 or any(source % resolution != 0 for resolution in requested):
        raise ValueError("all resolutions must divide source_resolution")
    master = rasterize_full_cell(structure, (source, source))
    grids = {
        resolution: (
            master.copy()
            if resolution == source
            else block_mean_downsample(master, resolution).astype(np.float32)
        )
        for resolution in requested
    }
    return FullCellRasterPyramid(source, grids, strategy)


__all__ = [
    "DIRECT_CELL_CENTER_V1",
    "MASTER_BLOCK_MEAN_V1",
    "FullCellRasterPyramid",
    "block_mean_downsample",
    "build_full_cell_raster_pyramid",
    "rasterize_full_cell",
]
