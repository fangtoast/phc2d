"""Physical reciprocal-space grids for full-cell datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np


def build_full_bz_grid(n_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a centered, duplicate-free square-BZ grid in reduced coordinates."""

    n = int(n_side)
    if n <= 0 or n % 2 == 0:
        raise ValueError("full BZ n_side must be a positive odd integer")
    half = n // 2
    indices = np.asarray(
        [
            (i, j)
            for i in range(-half, half + 1)
            for j in range(-half, half + 1)
        ],
        dtype=np.int64,
    )
    points = indices.astype(np.float64) / float(n)
    return points, indices


__all__ = ["build_full_bz_grid"]
