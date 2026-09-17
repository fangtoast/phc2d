"""Materialize equal-domain full-cell train/validation without test labels.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_mixed_validation(
    sources: Mapping[str, Mapping[str, Any]],
    output: str | Path,
    *,
    train_per_domain: int,
    validation_per_domain: int,
) -> dict[str, Any]:
    """Copy exactly two disjoint source splits with domain/source-index lineage.

    Source test indices are audited, but test grids/targets are never loaded.
    All source train rows are required: no additional subset-selection variable.
    """
    import h5py

    if len(sources) != 2 or min(train_per_domain, validation_per_domain) <= 0:
        raise ValueError("mixed contract requires two domains and positive split sizes")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    contracts = []
    shared_points = None
    shared_shapes = None
    for domain, spec in sources.items():
        path = Path(spec["path"])
        actual = file_sha256(path)
        if actual != spec["sha256"]:
            raise RuntimeError(f"source hash mismatch: {domain}")
        with h5py.File(path, "r") as h:
            indices = {s: np.sort(np.asarray(h[f"idx_{s}"], dtype=np.int64)) for s in ("train", "val", "test")}
            for s, rows in indices.items():
                if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(h['Wh'])):
                    raise ValueError(f"invalid split indices: {domain}:{s}")
            if any(np.intersect1d(indices[a], indices[b]).size for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
                raise ValueError(f"source split overlap: {domain}")
            if len(indices["train"]) != train_per_domain or len(indices["val"]) != validation_per_domain:
                raise ValueError(f"source split size mismatch: {domain}")
            grid = str(spec["grid_field"])
            shapes = (tuple(h[grid].shape[1:]), tuple(h["Wh"].shape[1:]))
            points = np.asarray(h["k_points"], dtype=np.float32)
            if len(shapes[0]) != 2 or len(shapes[1]) != 2 or points.shape != (shapes[1][0], 2):
                raise ValueError(f"invalid full-cell shapes: {domain}")
            if shared_shapes is not None and (shapes != shared_shapes or not np.array_equal(points, shared_points)):
                raise ValueError("mixed source shape/k-point mismatch")
            shared_shapes, shared_points = shapes, points
            contracts.append((domain, spec, indices))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    total = 2 * (train_per_domain + validation_per_domain)
    lineage = []
    with h5py.File(temporary, "x") as h:
        h.attrs["schema_version"] = "phc2d.full_cell_mixed_train_validation.v1"
        h.attrs["test_targets_loaded"] = False
        h.create_dataset("input_grid", (total, *shared_shapes[0]), dtype="f4", chunks=(min(16, total), *shared_shapes[0]), compression="lzf")
        h.create_dataset("Wh", (total, *shared_shapes[1]), dtype="f4")
        h.create_dataset("k_points", data=shared_points)
        h.create_dataset("idx_train", data=np.arange(2 * train_per_domain, dtype=np.int64))
        h.create_dataset("idx_val", data=np.arange(2 * train_per_domain, total, dtype=np.int64))
        h.create_dataset("idx_test", data=np.empty(0, dtype=np.int64))
        domains = h.create_dataset("source_domain_index", (total,), dtype="i8")
        original = h.create_dataset("source_row_index", (total,), dtype="i8")
        offset = 0
        for split in ("train", "val"):
            for domain_index, (domain, spec, indices) in enumerate(contracts):
                selected = indices[split]
                with h5py.File(spec["path"], "r") as source:
                    for start in range(0, len(selected), 64):
                        rows = selected[start:start + 64]
                        target = slice(offset + start, offset + start + len(rows))
                        h["input_grid"][target] = source[spec["grid_field"]][rows]
                        h["Wh"][target] = source["Wh"][rows]
                domains[offset:offset + len(selected)] = domain_index
                original[offset:offset + len(selected)] = selected
                lineage.append({"domain": domain, "split": split, "count": len(selected), "mixed_start": offset,
                                "source_sha256": spec["sha256"], "source_grid_field": spec["grid_field"],
                                "source_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest()})
                offset += len(selected)
        h.attrs["domain_order_json"] = json.dumps(list(sources))
        h.attrs["lineage_json"] = json.dumps(lineage)
    temporary.rename(destination)
    return {"path": str(destination), "sha256": file_sha256(destination), "grid_field": "input_grid",
            "grid_shape": list(shared_shapes[0]), "target_shape": list(shared_shapes[1]),
            "n_train": 2 * train_per_domain, "n_val": 2 * validation_per_domain,
            "test_targets_loaded": False, "lineage": lineage}
