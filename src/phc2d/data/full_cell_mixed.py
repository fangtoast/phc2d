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


def resolve_nested_train_subset(
    spec: Mapping[str, Any], *, parent_size: int, selected_size: int, salt: str,
    expected_parent_sha256: str,
) -> dict[str, Any]:
    """Resolve source row identities inside an independently audited parent subset.

    Only split metadata and training sample IDs are read; target labels are not.
    The parent digest must come from the relevant historical training evidence.
    """
    import h5py

    from phc2d.data.sampling import select_nested_sample_indices

    path = Path(spec["path"])
    if file_sha256(path) != spec["sha256"]:
        raise ValueError("source dataset hash mismatch")
    if not salt or not expected_parent_sha256 or not 0 < selected_size <= parent_size:
        raise ValueError("invalid nested subset contract")
    with h5py.File(path, "r") as handle:
        train = np.sort(np.asarray(handle["idx_train"], dtype=np.int64))
        if (train.ndim != 1 or len(np.unique(train)) != len(train)
                or np.any(train < 0) or np.any(train >= len(handle["Wh"]))
                or len(train) < parent_size):
            raise ValueError("invalid or insufficient training split")
        ids = np.asarray(handle["sample_id"][train]).astype("U").tolist()
        if len(set(ids)) != len(ids):
            raise ValueError("training sample IDs are not unique")
        parent = (train if len(train) == parent_size else
                  select_nested_sample_indices(train, ids, size=parent_size, salt=salt))
        if hashlib.sha256(parent.tobytes()).hexdigest() != expected_parent_sha256:
            raise ValueError("historical parent subset hash mismatch")
        parent_ids = np.asarray(handle["sample_id"][parent]).astype("U").tolist()
        selected = select_nested_sample_indices(parent, parent_ids, size=selected_size, salt=salt)
    return dict(parent_train_indices=parent, parent_train_indices_sha256=expected_parent_sha256,
                selected_train_indices=selected,
                selected_train_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                selection_rule="sha256-sample-id-nested-v1", selection_salt=salt)


def materialize_nested_mixed_validation(
    sources: Mapping[str, Mapping[str, Any]], output: str | Path, *,
    parent_size: int, selected_size: int, validation_per_domain: int, salt: str,
    expected_parent_hashes: Mapping[str, str], expected_selected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Materialize a two-domain dataset only after both nested identities match."""
    if len(sources) != 2 or set(sources) != set(expected_parent_hashes) or set(sources) != set(expected_selected_hashes):
        raise ValueError("nested source and expected identity domains differ")
    resolved = {}
    for domain, source in sources.items():
        selection = resolve_nested_train_subset(source, parent_size=parent_size, selected_size=selected_size,
                                                salt=salt, expected_parent_sha256=expected_parent_hashes[domain])
        if selection["selected_train_indices_sha256"] != expected_selected_hashes[domain]:
            raise ValueError(f"selected train subset hash mismatch: {domain}")
        resolved[domain] = {**source, **selection}
    return materialize_mixed_validation(resolved, output, train_per_domain=selected_size,
                                        validation_per_domain=validation_per_domain)


def materialize_mixed_validation(
    sources: Mapping[str, Mapping[str, Any]],
    output: str | Path,
    *,
    train_per_domain: int,
    validation_per_domain: int,
) -> dict[str, Any]:
    """Copy exactly two disjoint source splits with domain/source-index lineage.

    Source test indices are audited, but test grids/targets are never loaded.
    By default all source train rows are required. A source may instead provide
    explicit ``parent_train_indices``, its independently frozen
    ``parent_train_indices_sha256``, and ``selected_train_indices``. Both are
    original source row indices, not mixed-file positions.
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
            subset_keys = {"parent_train_indices", "parent_train_indices_sha256", "selected_train_indices"}
            present = subset_keys.intersection(spec)
            if present and present != subset_keys:
                raise ValueError(f"incomplete explicit subset contract: {domain}")
            if present:
                parent = np.asarray(spec["parent_train_indices"])
                selected = np.asarray(spec["selected_train_indices"])
                if (parent.ndim != 1 or selected.ndim != 1
                        or parent.dtype.kind not in "iu" or selected.dtype.kind not in "iu"):
                    raise ValueError(f"subset indices must be integer vectors: {domain}")
                parent = np.sort(parent.astype(np.int64))
                selected = np.sort(selected.astype(np.int64))
                if (len(np.unique(parent)) != len(parent) or len(np.unique(selected)) != len(selected)
                        or len(selected) != train_per_domain or not len(parent)
                        or not np.isin(parent, indices["train"]).all()
                        or not np.isin(selected, parent).all()):
                    raise ValueError(f"invalid nested train subset: {domain}")
                if hashlib.sha256(parent.tobytes()).hexdigest() != spec["parent_train_indices_sha256"]:
                    raise ValueError(f"parent train subset hash mismatch: {domain}")
                indices["train"] = selected
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
                if split == "train" and "parent_train_indices_sha256" in spec:
                    lineage[-1]["parent_train_indices_sha256"] = spec["parent_train_indices_sha256"]
                    lineage[-1]["selection_rule"] = spec.get("selection_rule", "explicit-source-row-indices")
                    lineage[-1]["selection_salt"] = spec.get("selection_salt")
                offset += len(selected)
        h.attrs["domain_order_json"] = json.dumps(list(sources))
        h.attrs["lineage_json"] = json.dumps(lineage)
    temporary.rename(destination)
    return {"path": str(destination), "sha256": file_sha256(destination), "grid_field": "input_grid",
            "grid_shape": list(shared_shapes[0]), "target_shape": list(shared_shapes[1]),
            "n_train": 2 * train_per_domain, "n_val": 2 * validation_per_domain,
            "test_targets_loaded": False, "lineage": lineage}
