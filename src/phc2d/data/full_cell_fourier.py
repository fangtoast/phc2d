"""Reusable Fourier and periodic-grid primitives for full-cell structures.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def periodic_cell_centers(grid_shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-centred periodic coordinates for a two-dimensional grid."""

    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain two dimensions")
    rows, cols = (int(grid_shape[0]), int(grid_shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape dimensions must be positive")
    x = -0.5 + (np.arange(cols, dtype=np.float64) + 0.5) / float(cols)
    y = -0.5 + (np.arange(rows, dtype=np.float64) + 0.5) / float(rows)
    return np.meshgrid(x, y, indexing="xy")


def evaluate_real_fourier_field(
    x: np.ndarray,
    y: np.ndarray,
    modes: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Evaluate ordered real Fourier modes using float64 broadcasting."""

    x_arr, y_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
    )
    result = np.zeros(x_arr.shape, dtype=np.float64)
    for row in modes:
        phase = 2.0 * np.pi * (
            int(row["mx"]) * x_arr + int(row["my"]) * y_arr
        )
        result += float(row.get("cos", 0.0)) * np.cos(phase)
        result += float(row.get("sin", 0.0)) * np.sin(phase)
    return result


def independent_half_plane_modes(cutoff: int) -> tuple[tuple[int, int], ...]:
    """Return the deterministic half-plane mode set inside a hard radial cutoff."""

    limit = int(cutoff)
    if limit <= 0:
        raise ValueError("cutoff must be positive")
    rows = [
        (mx, my)
        for mx in range(-limit, limit + 1)
        for my in range(-limit, limit + 1)
        if (mx > 0 or (mx == 0 and my > 0))
        and mx * mx + my * my <= limit * limit
    ]
    return tuple(rows)


def wrapped_periodic_delta(value: np.ndarray, center: float) -> np.ndarray:
    """Return the signed shortest displacement on a unit-periodic axis."""

    return np.mod(
        np.asarray(value, dtype=np.float64) - float(center) + 0.5,
        1.0,
    ) - 0.5


__all__ = [
    "evaluate_real_fourier_field",
    "independent_half_plane_modes",
    "periodic_cell_centers",
    "wrapped_periodic_delta",
]
