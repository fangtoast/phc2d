"""p4m wedge lifting utilities for square photonic-crystal grids.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


GridShape = tuple[int, int]


def infer_square_grid_shape_from_wedge_length(wedge_length: int) -> GridShape:
    """Infer a square p4m grid shape from triangular wedge vector length."""

    if wedge_length <= 0:
        raise ValueError("wedge_length must be positive")
    half_size = int(((8 * int(wedge_length) + 1) ** 0.5 - 1) / 2)
    if half_size * (half_size + 1) // 2 != int(wedge_length):
        raise ValueError(
            "Cannot infer square p4m grid shape from wedge length "
            f"{wedge_length}; pass grid_shape explicitly."
        )
    full_size = half_size * 2
    return (full_size, full_size)


def _validate_square_even_grid(grid_shape: GridShape) -> GridShape:
    rows, cols = (int(grid_shape[0]), int(grid_shape[1]))
    if rows <= 0 or cols <= 0 or rows != cols or rows % 2:
        raise ValueError(f"p4m wedge lifting requires a positive even square grid, got {grid_shape}")
    return (rows, cols)


@lru_cache(maxsize=None)
def wedge_indices(grid_shape: GridShape) -> tuple[tuple[int, int], ...]:
    """Return triangular p4m wedge pixel coordinates for ``grid_shape``."""

    rows, cols = _validate_square_even_grid(grid_shape)
    half = rows // 2
    w: list[tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            if i >= half and j >= half and j >= i:
                w.append((i, j))
    return tuple(w)


def _symmetries_for_grid(grid_shape: GridShape):
    """Return D4 symmetry operations for an even square grid."""

    rows, _ = _validate_square_even_grid(grid_shape)
    last = float(rows - 1)
    def ident(rc: np.ndarray) -> np.ndarray:
        return rc.copy()

    def rot90(rc: np.ndarray) -> np.ndarray:
        r, c = float(rc[0, 0]), float(rc[1, 0])
        return np.array([[c], [last - r]])

    def rot180(rc: np.ndarray) -> np.ndarray:
        r, c = float(rc[0, 0]), float(rc[1, 0])
        return np.array([[last - r], [last - c]])

    def rot270(rc: np.ndarray) -> np.ndarray:
        r, c = float(rc[0, 0]), float(rc[1, 0])
        return np.array([[last - c], [r]])

    def refl_h(rc: np.ndarray) -> np.ndarray:
        r, c = float(rc[0, 0]), float(rc[1, 0])
        return np.array([[r], [last - c]])

    def refl_v(rc: np.ndarray) -> np.ndarray:
        r, c = float(rc[0, 0]), float(rc[1, 0])
        return np.array([[last - r], [c]])

    def swap(rc: np.ndarray) -> np.ndarray:
        r, c = float(rc[0, 0]), float(rc[1, 0])
        return np.array([[c], [r]])

    def swap_m(rc: np.ndarray) -> np.ndarray:
        r, c = float(rc[0, 0]), float(rc[1, 0])
        return np.array([[last - c], [last - r]])

    return [ident, rot90, rot180, rot270, refl_h, refl_v, swap, swap_m]


@lru_cache(maxsize=None)
def _symmetries(grid_shape: GridShape):
    return tuple(_symmetries_for_grid(grid_shape))


def lift_rho_to_grid(rho_wedge: np.ndarray, grid_shape: GridShape | None = None) -> np.ndarray:
    """Map one p4m wedge vector to a full square occupancy grid."""

    flat = np.asarray(rho_wedge, dtype=np.float64).reshape(-1)
    resolved_grid_shape = (
        _validate_square_even_grid(grid_shape)
        if grid_shape is not None
        else infer_square_grid_shape_from_wedge_length(int(flat.shape[0]))
    )
    wedge = wedge_indices(resolved_grid_shape)
    if flat.shape[0] != len(wedge):
        raise ValueError(
            f"rho_wedge length {flat.shape[0]} does not match grid_shape "
            f"{resolved_grid_shape} wedge length {len(wedge)}"
        )
    grid = np.zeros(resolved_grid_shape, dtype=np.float64)
    for w_idx, (iw, jw) in enumerate(wedge):
        v = flat[w_idx]
        rc0 = np.array([[float(iw)], [float(jw)]])
        for f in _symmetries(resolved_grid_shape):
            rc = f(rc0)
            ii = int(np.clip(rc[0, 0], 0, resolved_grid_shape[0] - 1))
            jj = int(np.clip(rc[1, 0], 0, resolved_grid_shape[1] - 1))
            grid[ii, jj] = v
    return grid


def epsilon_from_rho(
    rho_wedge: np.ndarray,
    eps_air: float = 1.0,
    eps_alum: float = 8.9,
    grid_shape: GridShape | None = None,
) -> np.ndarray:
    """Map one p4m wedge vector to a piecewise-constant epsilon grid."""

    chi = lift_rho_to_grid(rho_wedge, grid_shape=grid_shape)
    return eps_air + (eps_alum - eps_air) * chi


def wedge_index_for_pixel(i: int, j: int, grid_shape: GridShape) -> int | None:
    """Return the wedge index for a pixel if the pixel lies in the p4m wedge."""

    wedge_to_idx = {p: k for k, p in enumerate(wedge_indices(_validate_square_even_grid(grid_shape)))}
    return wedge_to_idx.get((int(i), int(j)))


__all__ = [
    "GridShape",
    "epsilon_from_rho",
    "infer_square_grid_shape_from_wedge_length",
    "lift_rho_to_grid",
    "wedge_index_for_pixel",
    "wedge_indices",
]
