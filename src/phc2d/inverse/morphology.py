"""Morphology metrics for inverse-design candidates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def periodic_binary_offset(grid: Any, *, offset: float) -> np.ndarray:
    """Offset binary material on a unit square torus using cell-center distances.

    Positive offsets dilate material, negative offsets erode it; units are
    fractions of the lattice constant. This is a grid approximation, with a
    resolution-dependent boundary error; it is not an analytic geometric offset.
    The returned bool array retains the input resolution.
    """
    from scipy.ndimage import distance_transform_edt

    array = np.asarray(grid)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise ValueError("grid must be a nonempty square binary array")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError("grid must contain only finite binary values")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    material = array.astype(bool)
    if offset == 0 or not material.any() or material.all():
        return material.copy()
    size = material.shape[0]
    tiled = np.tile(material, (3, 3))
    # Distance to the opposite phase on a 3x3 tiling gives toroidal distances
    # for every center-cell pixel, including pixels across periodic boundaries.
    distance = distance_transform_edt(~tiled if offset > 0 else tiled,
                                      sampling=1.0 / size)[size:2*size, size:2*size]
    if offset > 0:
        return material | (distance <= offset)
    return material & (distance > -offset)


def _connected_components_4(binary: np.ndarray) -> list[int]:
    """Return 4-neighbor component sizes for a 2D binary grid."""

    h, w = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    sizes: list[int] = []
    for row in range(h):
        for col in range(w):
            if binary[row, col] == 0 or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] == 1 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


def binary_morphology_metrics(grid: Any, *, threshold: float = 0.5) -> dict[str, float]:
    """Return shape metrics for one binary or soft grid."""

    g = (np.asarray(grid, dtype=np.float32) >= float(threshold)).astype(np.uint8)
    if g.ndim == 1:
        g = g.reshape(1, g.shape[0])
    elif g.ndim != 2:
        g = g.reshape(g.shape[-2], g.shape[-1])
    fill = float(g.mean())
    gint = g.astype(np.int16)
    right = float(np.abs(gint[:, 1:] - gint[:, :-1]).sum()) if g.shape[1] > 1 else 0.0
    down = float(np.abs(gint[1:, :] - gint[:-1, :]).sum()) if g.shape[0] > 1 else 0.0
    total_variation = right + down

    sizes = _connected_components_4(g)
    n = len(sizes)
    largest_component_ratio = float(max(sizes) / max(int(g.sum()), 1)) if sizes else 0.0
    min_feature_size = float(min(sizes)) if sizes else 0.0

    neigh = np.zeros_like(g, dtype=np.int32)
    neigh[1:, :] += g[:-1, :]
    neigh[:-1, :] += g[1:, :]
    neigh[:, 1:] += g[:, :-1]
    neigh[:, :-1] += g[:, 1:]
    isolated = int(np.sum((g == 1) & (neigh == 0))) if g.sum() > 0 else 0

    return {
        "fill_fraction": fill,
        "num_connected_components": float(n),
        "largest_component_ratio": float(largest_component_ratio),
        "min_feature_size": float(min_feature_size),
        "isolated_pixel_count": float(isolated),
        "total_variation": float(total_variation),
    }


def candidate_morphology_metrics(candidates: Any, *, threshold: float = 0.5) -> dict[str, float]:
    """Aggregate simplified morphology metrics across a candidate tensor."""

    arr = np.asarray(candidates, dtype=np.float32)
    if arr.shape[0] == 0:
        return {
            "mean_fill_fraction": 0.0,
            "mean_connected_components": 0.0,
            "mean_isolated_pixel_count": 0.0,
            "mean_total_variation": 0.0,
            "mean_morphology_score": 0.0,
        }
    per = [binary_morphology_metrics(item, threshold=threshold) for item in arr]
    scores = [morphology_score(metrics) for metrics in per]
    return {
        "mean_fill_fraction": float(np.mean([item["fill_fraction"] for item in per])),
        "mean_connected_components": float(np.mean([item["num_connected_components"] for item in per])),
        "mean_isolated_pixel_count": float(np.mean([item["isolated_pixel_count"] for item in per])),
        "mean_total_variation": float(np.mean([item["total_variation"] for item in per])),
        "mean_morphology_score": float(np.mean(scores)),
    }


def morphology_score(metrics: dict[str, float]) -> float:
    """Compact validity score in [0, 1] for ranking candidates."""

    fill = metrics.get("fill_fraction", 0.0)
    fill_penalty = abs(fill - 0.5) * 2.0
    component_penalty = max(0.0, metrics.get("num_connected_components", 0.0) - 1.0) * 0.05
    isolated_penalty = metrics.get("isolated_pixel_count", 0.0) * 0.02
    score = 1.0 - min(1.0, fill_penalty + component_penalty + isolated_penalty)
    return float(max(0.0, min(1.0, score)))


__all__ = ["binary_morphology_metrics", "candidate_morphology_metrics", "morphology_score"]
