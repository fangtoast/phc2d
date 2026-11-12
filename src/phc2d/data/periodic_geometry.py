"""Connectivity of periodic binary cells.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def periodic_component_sizes(binary: Any) -> tuple[int, ...]:
    """Return four-neighbor component areas in pixels on a 2D torus."""
    mask = np.asarray(binary, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('periodic connectivity expects a 2D grid')
    rows, cols = mask.shape
    parent: dict[int, int] = {}

    def root(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        a_root, b_root = root(a), root(b)
        if a_root != b_root:
            parent[b_root] = a_root

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col]:
                continue
            current = row * cols + col
            parent.setdefault(current, current)
            for nr, nc in ((row, (col + 1) % cols), ((row + 1) % rows, col)):
                if mask[nr, nc]:
                    union(current, nr * cols + nc)
    counts: dict[int, int] = {}
    for value in parent:
        representative = root(value)
        counts[representative] = counts.get(representative, 0) + 1
    return tuple(sorted(counts.values()))
