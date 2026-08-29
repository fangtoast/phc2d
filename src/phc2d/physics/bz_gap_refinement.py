"""Adaptive Brillouin-zone sampling primitives for adjacent-band gap audits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from phc2d.forward.full_cell_metrics import enumerate_adjacent_gaps


def wrap_reduced_k_points(points: np.ndarray) -> np.ndarray:
    """Wrap reduced reciprocal coordinates to the centered first-BZ torus."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"points must be finite with shape [n,2], got {values.shape}")
    return np.mod(values + 0.5, 1.0) - 0.5


def _point_key(point: Sequence[float], *, decimals: int = 12) -> tuple[float, float]:
    values = np.asarray(point, dtype=np.float64).reshape(-1)
    if len(values) != 2:
        raise ValueError("one k point must contain two coordinates")
    return tuple(float(value) for value in np.round(values, decimals))


def propose_gap_extrema_refinement_points(
    k_points: np.ndarray,
    bands: np.ndarray,
    *,
    step: float,
    lower_band_min_one_based: int = 1,
    lower_band_max_one_based: int | None = None,
) -> np.ndarray:
    """Propose unseen 3x3 torus stencils around every adjacent-gap edge."""

    points = wrap_reduced_k_points(k_points)
    values = np.asarray(bands, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(points):
        raise ValueError("bands must have shape [n_k_points,n_bands] aligned with k_points")
    if not np.isfinite(values).all() or float(step) <= 0.0:
        raise ValueError("bands must be finite and step must be positive")
    gaps = enumerate_adjacent_gaps(
        values,
        lower_band_min_one_based=lower_band_min_one_based,
        lower_band_max_one_based=lower_band_max_one_based,
    )
    centers = []
    for gap in gaps:
        centers.extend(
            (points[gap.lower_edge_k_index], points[gap.upper_edge_k_index])
        )
    offsets = np.asarray(
        [(dx, dy) for dx in (-step, 0.0, step) for dy in (-step, 0.0, step)],
        dtype=np.float64,
    )
    existing = {_point_key(point) for point in points}
    proposed: dict[tuple[float, float], np.ndarray] = {}
    for center in centers:
        candidates = wrap_reduced_k_points(center[None, :] + offsets)
        for candidate in candidates:
            key = _point_key(candidate)
            if key not in existing:
                proposed.setdefault(key, candidate)
    if not proposed:
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([proposed[key] for key in sorted(proposed)], axis=0)


def adjacent_gap_records(
    bands: np.ndarray,
    *,
    lower_band_min_one_based: int = 1,
    lower_band_max_one_based: int | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-compatible adjacent-gap records for one sampled BZ."""

    return [
        asdict(gap)
        for gap in enumerate_adjacent_gaps(
            bands,
            lower_band_min_one_based=lower_band_min_one_based,
            lower_band_max_one_based=lower_band_max_one_based,
        )
    ]


def compare_gap_records(
    coarse: Sequence[dict[str, Any]],
    refined: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare aligned coarse/refined adjacent gaps without selecting one pair."""

    coarse_by_band = {int(row["lower_band_one_based"]): row for row in coarse}
    refined_by_band = {int(row["lower_band_one_based"]): row for row in refined}
    if coarse_by_band.keys() != refined_by_band.keys():
        raise ValueError("coarse and refined gap records cover different band pairs")
    rows: list[dict[str, Any]] = []
    for lower_band in sorted(coarse_by_band):
        left = coarse_by_band[lower_band]
        right = refined_by_band[lower_band]
        coarse_width = float(left["signed_width"])
        refined_width = float(right["signed_width"])
        rows.append(
            {
                "lower_band_one_based": lower_band,
                "upper_band_one_based": lower_band + 1,
                "coarse_lower_edge_frequency": float(left["lower_edge_frequency"]),
                "coarse_upper_edge_frequency": float(left["upper_edge_frequency"]),
                "coarse_center": float(left["center"]),
                "coarse_signed_width": coarse_width,
                "coarse_status": str(left["status"]),
                "refined_lower_edge_frequency": float(right["lower_edge_frequency"]),
                "refined_upper_edge_frequency": float(right["upper_edge_frequency"]),
                "refined_center": float(right["center"]),
                "refined_signed_width": refined_width,
                "refined_status": str(right["status"]),
                "gap_width_change": refined_width - coarse_width,
                "coarse_gap_overestimate": coarse_width - refined_width,
                "center_absolute_change": abs(float(right["center"]) - float(left["center"])),
                "status_changed": str(left["status"]) != str(right["status"]),
            }
        )
    return rows


__all__ = [
    "adjacent_gap_records",
    "compare_gap_records",
    "propose_gap_extrema_refinement_points",
    "wrap_reduced_k_points",
]
