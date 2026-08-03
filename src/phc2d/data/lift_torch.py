"""Torch implementation of p4m wedge lifting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from phc2d.data.lift import GridShape, infer_square_grid_shape_from_wedge_length, lift_rho_to_grid


@lru_cache(maxsize=None)
def _wedge_assignment(grid_shape: GridShape, wedge_length: int) -> tuple[tuple[int, int, int], ...]:
    """Return ``(wedge_index, row, col)`` assignments for a p4m lift."""

    import numpy as np

    out: list[tuple[int, int, int]] = []
    for idx in range(int(wedge_length)):
        one = np.zeros(int(wedge_length), dtype=np.float32)
        one[idx] = 1.0
        grid = lift_rho_to_grid(one, grid_shape=grid_shape)
        for row, col in zip(*np.where(grid > 0.5), strict=False):
            out.append((idx, int(row), int(col)))
    return tuple(out)


def _resolve_grid_shape(grid_shape: GridShape | None, wedge_length: int) -> GridShape:
    if grid_shape is None:
        return infer_square_grid_shape_from_wedge_length(int(wedge_length))
    return (int(grid_shape[0]), int(grid_shape[1]))


def lift_rho_to_grid_torch(rho: Any, *, grid_shape: GridShape | None = None) -> Any:
    """Map ``(..., wedge_length)`` tensors to ``(..., 1, height, width)`` grids."""

    import torch

    x = torch.as_tensor(rho)
    if len(x.shape) == 0:
        raise ValueError("rho must have a wedge dimension")
    wedge_length = int(x.shape[-1])
    resolved_grid_shape = _resolve_grid_shape(grid_shape, wedge_length)
    flat = x.reshape(-1, wedge_length)
    out = flat.new_zeros((flat.shape[0], 1, resolved_grid_shape[0], resolved_grid_shape[1]))
    for wedge_idx, row, col in _wedge_assignment(resolved_grid_shape, wedge_length):
        out[:, 0, row, col] = flat[:, wedge_idx]
    return out.reshape(*x.shape[:-1], 1, resolved_grid_shape[0], resolved_grid_shape[1])


def epsilon_from_rho_torch(
    rho: Any,
    *,
    eps_air: float = 1.0,
    eps_alum: float = 8.9,
    grid_shape: GridShape | None = None,
) -> Any:
    """Map p4m wedge tensors to differentiable epsilon grids."""

    grid = lift_rho_to_grid_torch(rho, grid_shape=grid_shape)
    return float(eps_air) + (float(eps_alum) - float(eps_air)) * grid


__all__ = ["epsilon_from_rho_torch", "lift_rho_to_grid_torch"]
