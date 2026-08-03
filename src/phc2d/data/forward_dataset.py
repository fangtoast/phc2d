"""Forward dataset load/save helpers with one stable NPZ schema.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ForwardDatasetBundle:
    """Arrays and metadata for a ``rho -> Wh`` forward-surrogate dataset."""

    rho: np.ndarray
    Wh: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    config_yaml: str = ""
    metadata: list[dict[str, Any]] = field(default_factory=list)
    dataset_path: Path | None = None

    @property
    def nk(self) -> int:
        return int(self.Wh.shape[1])

    @property
    def nb(self) -> int:
        return int(self.Wh.shape[2])

    @property
    def n_samples(self) -> int:
        return int(self.rho.shape[0])


def _default_splits(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = max(1, int(0.8 * n))
    n_val = max(n_train, int(0.9 * n))
    return (
        np.arange(n_train, dtype=np.int64),
        np.arange(n_train, n_val, dtype=np.int64),
        np.arange(n_val, n, dtype=np.int64),
    )


def _read_scalar_text(data: Mapping[str, Any], key: str) -> str:
    if key not in data:
        return ""
    raw = data[key]
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def serialize_metadata_jsonl(metadata: list[dict[str, Any]]) -> str:
    """Serialize sample metadata as compact JSON Lines text."""
    return "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in metadata)


def parse_metadata_jsonl(raw: Any) -> list[dict[str, Any]]:
    """Parse metadata JSON Lines stored inside an NPZ scalar field."""
    if raw is None:
        return []
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _infer_split_labels(n: int, splits: Mapping[str, Any]) -> np.ndarray:
    labels = np.array(["unassigned"] * n, dtype=object)
    if n <= 0:
        return labels
    def _as_index_array(value: Any) -> np.ndarray:
        if value is None:
            return np.array([], dtype=np.int64)
        arr = np.asarray(value, dtype=np.int64)
        arr = arr.reshape(-1)
        return arr[(arr >= 0) & (arr < n)]

    for split_name, value in splits.items():
        for index in _as_index_array(value):
            labels[int(index)] = split_name
    return labels


def _coerce_metadata_rows(
    n_samples: int,
    dataset_path: Path,
    raw_metadata: list[dict[str, Any]],
    split_labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_name = dataset_path.stem
    for idx in range(n_samples):
        row = dict(raw_metadata[idx]) if idx < len(raw_metadata) and isinstance(raw_metadata[idx], dict) else {}
        source = str(row.get("raw_source") or row.get("source") or "")
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            sample_id = f"{dataset_name}_{idx:06d}"
            row["sample_id_source"] = "derived_from_index"
        else:
            row["sample_id_source"] = "raw"

        split = str(split_labels[idx]) if split_labels[idx] else "unassigned"
        row["split"] = split
        row["sample_id"] = sample_id
        row["dataset_id"] = dataset_name
        row["raw_source"] = source or "dataset_config"
        row.setdefault("generator_name", row.get("generator") )
        rows.append(row)
    return rows


def load_forward_dataset(path: str | Path) -> ForwardDatasetBundle:
    """Load a forward dataset and fill missing split fields using 80/10/10 defaults."""
    with np.load(Path(path), allow_pickle=False) as data:
        if "rho" not in data or "Wh" not in data:
            raise KeyError(f"Forward dataset must contain 'rho' and 'Wh': {path}")
        rho = np.asarray(data["rho"], dtype=np.float32)
        Wh = np.asarray(data["Wh"], dtype=np.float32)
        idx_train, idx_val, idx_test = _default_splits(int(rho.shape[0]))
        if "idx_train" in data:
            idx_train = np.asarray(data["idx_train"], dtype=np.int64)
        if "idx_val" in data:
            idx_val = np.asarray(data["idx_val"], dtype=np.int64)
        if "idx_test" in data:
            idx_test = np.asarray(data["idx_test"], dtype=np.int64)
        raw_metadata = parse_metadata_jsonl(data["metadata_jsonl"]) if "metadata_jsonl" in data else []
        split_labels = _infer_split_labels(
            int(rho.shape[0]),
            {
                "train": idx_train,
                "val": idx_val,
                "test": idx_test,
            },
        )
        metadata = _coerce_metadata_rows(int(rho.shape[0]), Path(path), raw_metadata, split_labels)
        config_yaml = _read_scalar_text(data, "config_yaml")
    return ForwardDatasetBundle(
        rho=rho,
        Wh=Wh,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        config_yaml=config_yaml,
        metadata=metadata,
        dataset_path=Path(path),
    )


def save_forward_dataset(path: str | Path, bundle: ForwardDatasetBundle, *, compressed: bool = True) -> Path:
    """Write a forward dataset using the shared NPZ schema."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rho": np.asarray(bundle.rho, dtype=np.float32),
        "Wh": np.asarray(bundle.Wh, dtype=np.float32),
        "idx_train": np.asarray(bundle.idx_train, dtype=np.int64),
        "idx_val": np.asarray(bundle.idx_val, dtype=np.int64),
        "idx_test": np.asarray(bundle.idx_test, dtype=np.int64),
        "config_yaml": str(bundle.config_yaml),
        "metadata_jsonl": serialize_metadata_jsonl(list(bundle.metadata)),
    }
    writer = np.savez_compressed if compressed else np.savez
    writer(out, **payload)
    return out
