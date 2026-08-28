"""Polarization-aware effective-medium and folded-frequency primitives.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


Polarization = Literal["TE", "TM"]


@dataclass(frozen=True)
class FoldedSquareLatticeFrequencies:
    """First-shell square-lattice frequencies in MPB ``a/lambda`` units."""

    gamma_g1: np.ndarray
    x: np.ndarray
    m: np.ndarray


def _normalize_spatial_axes(ndim: int, spatial_axes: Sequence[int]) -> tuple[int, ...]:
    axes = tuple(int(axis) % ndim for axis in spatial_axes)
    if not axes or len(set(axes)) != len(axes):
        raise ValueError("spatial_axes must contain unique axes")
    return axes


def effective_refractive_index(
    epsilon_grid: np.ndarray,
    polarization: str,
    *,
    spatial_axes: Sequence[int] = (-2, -1),
) -> np.ndarray:
    """Return the scalar zeroth-order effective index for TE or TM data.

    TE follows the harmonic-permittivity candidate
    ``1 / sqrt(mean(1 / epsilon))``. TM follows the arithmetic-permittivity
    candidate ``sqrt(mean(epsilon))``. The approximation is intentionally
    separate from real-MPB slope calibration.
    """

    epsilon = np.asarray(epsilon_grid, dtype=np.float64)
    if epsilon.ndim < 1:
        raise ValueError("epsilon_grid must expose at least one spatial axis")
    if not np.isfinite(epsilon).all():
        raise ValueError("epsilon_grid must be finite")
    if np.any(epsilon <= 0.0):
        raise ValueError("epsilon_grid must be strictly positive")
    axes = _normalize_spatial_axes(epsilon.ndim, spatial_axes)
    mode = str(polarization).upper()
    if mode == "TE":
        epsilon_effective = 1.0 / np.mean(1.0 / epsilon, axis=axes)
    elif mode == "TM":
        epsilon_effective = np.mean(epsilon, axis=axes)
    else:
        raise ValueError("polarization must be 'TE' or 'TM'")
    return np.sqrt(epsilon_effective)


def folded_square_lattice_frequencies(
    effective_index: np.ndarray | float,
) -> FoldedSquareLatticeFrequencies:
    """Return distinct Gamma-foldback, X-boundary, and M-boundary scales."""

    index = np.asarray(effective_index, dtype=np.float64)
    if not np.isfinite(index).all() or np.any(index <= 0.0):
        raise ValueError("effective_index must be finite and strictly positive")
    return FoldedSquareLatticeFrequencies(
        gamma_g1=1.0 / index,
        x=1.0 / (2.0 * index),
        m=np.sqrt(2.0) / (2.0 * index),
    )


__all__ = [
    "FoldedSquareLatticeFrequencies",
    "Polarization",
    "effective_refractive_index",
    "folded_square_lattice_frequencies",
]
