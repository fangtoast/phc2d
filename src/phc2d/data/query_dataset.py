"""Lazy band-query dataset contracts for query-form surrogate benchmarks.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.forward_dataset import ForwardDatasetBundle, load_forward_dataset
from phc2d.data.k_path import (
    expand_physical_holdout_indices,
    hash_k_path_spec,
    load_dataset_k_path_spec,
    physical_k_ids,
)
from phc2d.data.lift import epsilon_from_rho


@dataclass(frozen=True)
class KPathInfo:
    """Recovered k-path coordinates and their evidence boundary."""

    coordinates: np.ndarray
    source: str
    is_physical: bool
    has_segment_metadata: bool
    repeated_endpoint_indices: tuple[int, ...] = ()
    nodes: tuple[str, ...] = ()
    path_s: np.ndarray | None = None
    segment_id: np.ndarray | None = None
    segment_local_t: np.ndarray | None = None
    physical_k_id: tuple[str, ...] = ()
    k_path_id: str = ""
    k_path_hash: str = ""
    verification_status: str = ""

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[1]) if self.coordinates.ndim == 2 else 0


@dataclass(frozen=True)
class QueryProtocolSpec:
    """One query protocol with structure split and k-query masks."""

    name: str
    train_k_indices: tuple[int, ...]
    eval_k_indices: tuple[int, ...]
    holdout_kind: str
    description: str


@dataclass(frozen=True)
class BandQuerySpec:
    """Resolved query dataset configuration."""

    dataset_path: Path
    split: str
    protocol: QueryProtocolSpec
    input_repr: str = "epsilon_grid"
    normalize_query: bool = True


def _sha256(payload: Any) -> str:
    if isinstance(payload, bytes):
        data = payload
    else:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _parse_config_yaml(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    payload = yaml.safe_load(text) or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def recover_k_path(bundle: ForwardDatasetBundle) -> KPathInfo:
    """Recover k coordinates from dataset metadata, or fall back to normalized index coordinates.

    The fallback is deliberately labeled as non-physical because the NPZ forward
    datasets currently store `Wh` rows but no explicit Γ/X/M coordinates.
    """

    if bundle.dataset_path is not None:
        spec = load_dataset_k_path_spec(bundle.dataset_path)
        if spec is not None:
            if int(spec.Nk) != int(bundle.nk):
                raise ValueError(f"k-path sidecar Nk={spec.Nk} does not match dataset Wh Nk={bundle.nk}")
            coords = np.asarray(spec.k_points, dtype=np.float32)
            repeated = tuple(int(group[-1]) for group in spec.duplicate_coordinate_groups if len(group) > 1)
            return KPathInfo(
                coordinates=coords,
                source=f"sidecar:{spec.k_path_id}",
                is_physical=True,
                has_segment_metadata=bool(spec.segments),
                repeated_endpoint_indices=repeated,
                nodes=tuple(node.label for node in spec.nodes),
                path_s=np.asarray(spec.path_s, dtype=np.float32),
                segment_id=np.asarray(spec.segment_id, dtype=np.int64),
                segment_local_t=np.asarray(spec.segment_local_t, dtype=np.float32),
                physical_k_id=physical_k_ids(coords),
                k_path_id=spec.k_path_id,
                k_path_hash=hash_k_path_spec(spec),
                verification_status=spec.metadata.verification_status,
            )

    metadata = bundle.metadata[0] if bundle.metadata else {}
    for key in ("k_points", "k_coordinates", "k_path"):
        value = metadata.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == bundle.nk:
            repeated = _find_repeated_endpoint_indices(arr)
            return KPathInfo(
                coordinates=arr.astype(np.float32),
                source=f"metadata.{key}",
                is_physical=True,
                has_segment_metadata=bool(metadata.get("k_path_segments")),
                repeated_endpoint_indices=repeated,
                nodes=tuple(str(x) for x in metadata.get("k_path_nodes", ())),
                physical_k_id=physical_k_ids(arr),
            )

    config = _parse_config_yaml(bundle.config_yaml)
    k_cfg = config.get("k_path", {}) if isinstance(config.get("k_path"), Mapping) else {}
    for key in ("points", "coordinates", "k_points"):
        value = k_cfg.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == bundle.nk:
            repeated = _find_repeated_endpoint_indices(arr)
            return KPathInfo(
                coordinates=arr.astype(np.float32),
                source=f"config_yaml.k_path.{key}",
                is_physical=True,
                has_segment_metadata=bool(k_cfg.get("segments")),
                repeated_endpoint_indices=repeated,
                nodes=tuple(str(x) for x in k_cfg.get("nodes", ())),
                physical_k_id=physical_k_ids(arr),
            )

    n_total = int(k_cfg.get("n_total", bundle.nk)) if k_cfg else bundle.nk
    source = "normalized_index_from_Wh_shape"
    if n_total == bundle.nk:
        source = "normalized_index_from_config_yaml.k_path.n_total"
    coords = np.linspace(0.0, 1.0, bundle.nk, dtype=np.float32).reshape(bundle.nk, 1)
    return KPathInfo(
        coordinates=coords,
        source=source,
        is_physical=False,
        has_segment_metadata=False,
        repeated_endpoint_indices=(),
        nodes=(),
    )


def _find_repeated_endpoint_indices(coords: np.ndarray) -> tuple[int, ...]:
    repeated: list[int] = []
    for idx in range(1, int(coords.shape[0])):
        if np.allclose(coords[idx], coords[idx - 1]):
            repeated.append(idx)
    return tuple(repeated)


def build_query_masks(nk: int, *, has_segment_metadata: bool = False) -> dict[str, QueryProtocolSpec]:
    """Build deterministic Full-2A query protocols from the dataset k count."""

    all_k = tuple(range(int(nk)))
    internal = list(range(1, max(int(nk) - 1, 1)))
    if len(internal) >= 4:
        interp_holdout = tuple(internal[1::4] or internal[-1:])
    else:
        interp_holdout = tuple(internal[-1:])
    interp_train = tuple(idx for idx in all_k if idx not in set(interp_holdout))

    segment_size = max(1, int(round(0.2 * int(nk))))
    segment_holdout = tuple(range(max(0, int(nk) - segment_size), int(nk)))
    segment_train = tuple(idx for idx in all_k if idx not in set(segment_holdout))
    segment_name = "unseen_structure_heldout_segment" if has_segment_metadata else "contiguous_index_holdout"
    segment_kind = "physical_segment" if has_segment_metadata else "contiguous_index_holdout"

    return {
        "unseen_structure_all_k": QueryProtocolSpec(
            name="unseen_structure_all_k",
            train_k_indices=all_k,
            eval_k_indices=all_k,
            holdout_kind="none",
            description="Unseen structure prediction over the full available k path.",
        ),
        "unseen_structure_k_interpolation": QueryProtocolSpec(
            name="unseen_structure_k_interpolation",
            train_k_indices=interp_train,
            eval_k_indices=interp_holdout,
            holdout_kind="internal_index_holdout",
            description="Unseen structures evaluated on deterministic internal held-out k rows.",
        ),
        "unseen_structure_heldout_segment": QueryProtocolSpec(
            name=segment_name,
            train_k_indices=segment_train,
            eval_k_indices=segment_holdout,
            holdout_kind=segment_kind,
            description="Held-out path segment when physical segment metadata exists; otherwise contiguous index holdout.",
        ),
    }


def build_physical_query_masks(k_path: KPathInfo) -> dict[str, QueryProtocolSpec]:
    """Build Full-2A masks with duplicate physical coordinates held out together."""

    nk = int(k_path.coordinates.shape[0])
    all_k = tuple(range(nk))
    ids = k_path.physical_k_id or physical_k_ids(k_path.coordinates)
    node_indices: set[int] = {0, nk - 1}
    if k_path.segment_id is not None:
        changes = np.flatnonzero(np.diff(np.asarray(k_path.segment_id, dtype=np.int64)) != 0)
        for change in changes:
            node_indices.add(int(change) + 1)
    node_indices.update(int(i) for i in k_path.repeated_endpoint_indices)

    interior = [idx for idx in all_k if idx not in node_indices]
    interp_seed = tuple(interior[1::4] if len(interior) >= 4 else interior[-1:])
    interp_eval = expand_physical_holdout_indices(interp_seed, ids)
    interp_train = tuple(idx for idx in all_k if idx not in set(interp_eval))

    if k_path.segment_id is not None and len(set(int(v) for v in k_path.segment_id)) >= 2:
        held_segment = sorted(set(int(v) for v in k_path.segment_id))[-1]
        segment_seed = tuple(
            idx
            for idx, sid in enumerate(np.asarray(k_path.segment_id, dtype=np.int64))
            if int(sid) == held_segment and idx not in node_indices
        )
        segment_name = "unseen_structure_physical_heldout_segment"
        segment_kind = "physical_segment_interior"
    else:
        segment_size = max(1, int(round(0.2 * nk)))
        segment_seed = tuple(range(max(0, nk - segment_size), nk))
        segment_name = "contiguous_index_holdout"
        segment_kind = "contiguous_index_holdout"
    segment_eval = expand_physical_holdout_indices(segment_seed, ids)
    segment_train = tuple(idx for idx in all_k if idx not in set(segment_eval))

    return {
        "unseen_structure_all_k": QueryProtocolSpec(
            name="unseen_structure_all_k",
            train_k_indices=all_k,
            eval_k_indices=all_k,
            holdout_kind="none",
            description="Unseen structure prediction over the full physical k path.",
        ),
        "unseen_structure_k_interpolation": QueryProtocolSpec(
            name="unseen_structure_physical_k_interpolation",
            train_k_indices=interp_train,
            eval_k_indices=interp_eval,
            holdout_kind="physical_coordinate_interpolation",
            description="Internal physical k coordinates held out with duplicate coordinates moved together.",
        ),
        "unseen_structure_heldout_segment": QueryProtocolSpec(
            name=segment_name,
            train_k_indices=segment_train,
            eval_k_indices=segment_eval,
            holdout_kind=segment_kind,
            description="Physical segment interior holdout; endpoints are excluded from the main held-out mask.",
        ),
    }


class BandQueryDataset:
    """Lazy structure-plus-k query dataset returning one band vector per item."""

    def __init__(
        self,
        bundle: ForwardDatasetBundle,
        *,
        sample_indices: Sequence[int],
        k_indices: Sequence[int],
        k_path: KPathInfo,
        split: str,
        protocol: str,
        input_repr: str = "epsilon_grid",
    ) -> None:
        self.bundle = bundle
        self.sample_indices = tuple(int(i) for i in sample_indices)
        self.k_indices = tuple(int(i) for i in k_indices)
        self.k_path = k_path
        self.split = str(split)
        self.protocol = str(protocol)
        self.input_repr = str(input_repr)
        if self.input_repr != "epsilon_grid":
            raise ValueError("BandQueryDataset currently supports input_repr=epsilon_grid for Full-2A.")
        if not self.sample_indices:
            raise ValueError("sample_indices cannot be empty")
        if not self.k_indices:
            raise ValueError("k_indices cannot be empty")

    def __len__(self) -> int:
        return int(len(self.sample_indices) * len(self.k_indices))

    def _sample_id(self, sample_index: int, metadata: Mapping[str, Any]) -> str:
        sample_id = str(metadata.get("sample_id", "")).strip()
        if sample_id:
            return sample_id
        dataset_id = str(metadata.get("dataset_id", "query_dataset")).strip() or "query_dataset"
        return f"{dataset_id}_{int(sample_index):06d}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_pos = int(index) // len(self.k_indices)
        k_pos = int(index) % len(self.k_indices)
        sample_index = int(self.sample_indices[sample_pos])
        k_index = int(self.k_indices[k_pos])
        metadata = self.bundle.metadata[sample_index] if sample_index < len(self.bundle.metadata) else {}
        sample_id = self._sample_id(sample_index, metadata)
        return {
            "structure": epsilon_from_rho(self.bundle.rho[sample_index]).astype(np.float32),
            "k_coordinate": np.asarray(self.k_path.coordinates[k_index], dtype=np.float32),
            "k_index": np.asarray(k_index, dtype=np.int64),
            "path_s": np.asarray(0.0 if self.k_path.path_s is None else self.k_path.path_s[k_index], dtype=np.float32),
            "segment_id": np.asarray(-1 if self.k_path.segment_id is None else self.k_path.segment_id[k_index], dtype=np.int64),
            "segment_local_t": np.asarray(
                0.0 if self.k_path.segment_local_t is None else self.k_path.segment_local_t[k_index],
                dtype=np.float32,
            ),
            "physical_k_id": self.k_path.physical_k_id[k_index] if self.k_path.physical_k_id else f"pk_{k_index:04d}",
            "target": np.asarray(self.bundle.Wh[sample_index, k_index, :], dtype=np.float32),
            "sample_id": sample_id,
            "query_id": f"{sample_id}:{self.k_path.k_path_id or 'kpath_unknown'}:k_{k_index:04d}",
            "k_path_id": self.k_path.k_path_id,
            "k_path_hash": self.k_path.k_path_hash,
            "split": self.split,
            "protocol": self.protocol,
        }


def split_indices(bundle: ForwardDatasetBundle, split: str) -> np.ndarray:
    key = str(split).lower()
    if key == "train":
        return np.asarray(bundle.idx_train, dtype=np.int64)
    if key in {"val", "valid", "validation"}:
        return np.asarray(bundle.idx_val, dtype=np.int64)
    if key == "test":
        return np.asarray(bundle.idx_test, dtype=np.int64)
    raise ValueError(f"Unknown split: {split}")


def build_query_dataset(spec: BandQuerySpec) -> BandQueryDataset:
    bundle = load_forward_dataset(spec.dataset_path)
    k_path = recover_k_path(bundle)
    k_indices = spec.protocol.train_k_indices if spec.split == "train" else spec.protocol.eval_k_indices
    return BandQueryDataset(
        bundle,
        sample_indices=split_indices(bundle, spec.split),
        k_indices=k_indices,
        k_path=k_path,
        split=spec.split,
        protocol=spec.protocol.name,
        input_repr=spec.input_repr,
    )


def validate_query_split_integrity(
    bundle: ForwardDatasetBundle,
    protocols: Mapping[str, QueryProtocolSpec],
) -> dict[str, Any]:
    def _sid(index: int) -> str:
        if int(index) < len(bundle.metadata):
            row = bundle.metadata[int(index)]
            if row.get("sample_id"):
                return str(row["sample_id"])
            dataset_id = str(row.get("dataset_id", "query_dataset"))
            return f"{dataset_id}_{int(index):06d}"
        return f"query_dataset_{int(index):06d}"

    train_ids = {_sid(int(i)) for i in bundle.idx_train}
    val_ids = {_sid(int(i)) for i in bundle.idx_val}
    test_ids = {_sid(int(i)) for i in bundle.idx_test}
    protocol_payload = {}
    for name, protocol in protocols.items():
        train_k = set(protocol.train_k_indices)
        eval_k = set(protocol.eval_k_indices)
        protocol_payload[name] = {
            "train_k_count": len(train_k),
            "eval_k_count": len(eval_k),
            "heldout_k_absent_from_train": not bool(train_k & eval_k) if protocol.holdout_kind != "none" else None,
            "train_k_hash": _sha256(sorted(train_k)),
            "eval_k_hash": _sha256(sorted(eval_k)),
            "holdout_kind": protocol.holdout_kind,
        }
    return {
        "train_val_sample_id_disjoint": not bool(train_ids & val_ids),
        "train_test_sample_id_disjoint": not bool(train_ids & test_ids),
        "val_test_sample_id_disjoint": not bool(val_ids & test_ids),
        "train_sample_id_count": len(train_ids),
        "val_sample_id_count": len(val_ids),
        "test_sample_id_count": len(test_ids),
        "protocols": protocol_payload,
    }


def inspect_query_dataset(path: str | Path) -> dict[str, Any]:
    bundle = load_forward_dataset(path)
    k_path = recover_k_path(bundle)
    protocols = build_physical_query_masks(k_path) if k_path.is_physical else build_query_masks(bundle.nk, has_segment_metadata=k_path.has_segment_metadata)
    return {
        "dataset_path": str(path),
        "n_structures": bundle.n_samples,
        "Wh_shape": list(bundle.Wh.shape),
        "Wh_dtype": str(bundle.Wh.dtype),
        "Nk": bundle.nk,
        "Nb": bundle.nb,
        "splits": {
            "train": int(len(bundle.idx_train)),
            "val": int(len(bundle.idx_val)),
            "test": int(len(bundle.idx_test)),
        },
        "k_path": {
            "coordinate_source": k_path.source,
            "is_physical": k_path.is_physical,
            "dimension": k_path.dimension,
            "has_segment_metadata": k_path.has_segment_metadata,
            "repeated_endpoint_indices": list(k_path.repeated_endpoint_indices),
            "nodes": list(k_path.nodes),
            "k_path_id": k_path.k_path_id,
            "k_path_hash": k_path.k_path_hash,
            "verification_status": k_path.verification_status,
            "physical_k_id_count": len(set(k_path.physical_k_id)) if k_path.physical_k_id else None,
        },
        "sample_lineage": {
            "sample_id_available": all("sample_id" in row for row in bundle.metadata),
            "dataset_id": bundle.metadata[0].get("dataset_id") if bundle.metadata else None,
            "query_id_format": "<sample_id>:k_<zero_padded_k_index>",
        },
        "integrity": validate_query_split_integrity(bundle, protocols),
    }


def iter_query_ids(dataset: BandQueryDataset, *, limit: int | None = None) -> Iterable[str]:
    stop = len(dataset) if limit is None else min(int(limit), len(dataset))
    for idx in range(stop):
        yield str(dataset[idx]["query_id"])


__all__ = [
    "BandQueryDataset",
    "BandQuerySpec",
    "KPathInfo",
    "QueryProtocolSpec",
    "build_query_dataset",
    "build_query_masks",
    "build_physical_query_masks",
    "inspect_query_dataset",
    "iter_query_ids",
    "recover_k_path",
    "split_indices",
    "validate_query_split_integrity",
]
