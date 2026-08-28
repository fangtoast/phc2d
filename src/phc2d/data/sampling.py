"""Deterministic sample-selection contracts shared across workflows.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from collections.abc import Sequence

import numpy as np


def select_nested_sample_indices(
    indices: np.ndarray,
    sample_ids: Sequence[str],
    *,
    size: int,
    salt: str,
) -> np.ndarray:
    """Select a deterministic nested subset using stable sample-ID hashes."""

    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    ids = [str(value) for value in sample_ids]
    if len(ids) != int(values.size):
        raise ValueError("sample_ids must align one-to-one with indices")
    requested = int(size)
    if requested <= 0 or requested > int(values.size):
        raise ValueError(f"size must be in [1, {values.size}], got {requested}")
    ranked = sorted(
        zip(values.tolist(), ids, strict=True),
        key=lambda item: (
            hashlib.sha256(f"{salt}|{item[1]}".encode("utf-8")).digest(),
            int(item[0]),
        ),
    )
    return np.sort(
        np.asarray([item[0] for item in ranked[:requested]], dtype=np.int64)
    )


def select_stratified_sample_indices(
    indices: np.ndarray,
    strata: Sequence[Sequence[str]],
    stable_ids: Sequence[str],
    *,
    size: int,
) -> np.ndarray:
    """Round-robin deterministic rows across stable strata."""

    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    strata_rows = [tuple(str(value) for value in row) for row in strata]
    ids = [str(value) for value in stable_ids]
    if len(strata_rows) != int(values.size) or len(ids) != int(values.size):
        raise ValueError("strata and stable_ids must align one-to-one with indices")
    requested = int(size)
    if requested < 0 or requested > int(values.size):
        raise ValueError(f"size must be in [0, {values.size}], got {requested}")
    if requested == 0:
        return np.empty(0, dtype=np.int64)
    grouped: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for index, stratum, stable_id in zip(values.tolist(), strata_rows, ids, strict=True):
        grouped[stratum].append((stable_id, int(index)))
    queues = {
        key: deque(index for _, index in sorted(rows, key=lambda row: (row[0], row[1])))
        for key, rows in grouped.items()
    }
    selected: list[int] = []
    keys = sorted(queues)
    while len(selected) < requested:
        progressed = False
        for key in keys:
            if queues[key] and len(selected) < requested:
                selected.append(queues[key].popleft())
                progressed = True
        if not progressed:
            raise RuntimeError("stratified selection exhausted before reaching requested size")
    return np.sort(np.asarray(selected, dtype=np.int64))


__all__ = ["select_nested_sample_indices", "select_stratified_sample_indices"]
