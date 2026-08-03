"""Deterministic selection manifests for Bloch-eigenfield pilot datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.data.k_path import KPathSpec, physical_k_ids


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_rank_key(sample_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _balanced_counts(total: int, categories: int) -> list[int]:
    if total < 0 or categories <= 0:
        raise ValueError("total must be non-negative and categories must be positive.")
    base, extra = divmod(int(total), int(categories))
    return [base + (1 if index < extra else 0) for index in range(categories)]


def _select_split_structures(
    *,
    rho: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int],
    count: int,
    n_strata: int,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = []
    for index in source_indices:
        row = dict(metadata_rows[int(index)])
        sample_id = str(row.get("sample_id", f"source_{int(index):06d}"))
        candidates.append(
            {
                "source_sample_index": int(index),
                "source_sample_id": sample_id,
                "source_sample_hash": str(row.get("rho_hash", "")),
                "geometry_hash": row.get("geometry_hash"),
                "occupancy": float(np.sum(np.asarray(rho[int(index)]))),
                "stable_key": _stable_rank_key(sample_id, seed=seed),
            }
        )
    if int(count) > len(candidates):
        raise ValueError(f"Requested {count} {split} structures, but only {len(candidates)} are available.")
    candidates.sort(key=lambda item: (item["occupancy"], item["stable_key"]))
    selected: list[dict[str, Any]] = []
    stratum_counts = _balanced_counts(int(count), int(n_strata))
    population_bounds = np.linspace(0, len(candidates), int(n_strata) + 1, dtype=np.int64)
    for stratum, requested in enumerate(stratum_counts):
        start, stop = int(population_bounds[stratum]), int(population_bounds[stratum + 1])
        pool = sorted(candidates[start:stop], key=lambda item: item["stable_key"])
        if int(requested) > len(pool):
            raise ValueError(f"Occupancy stratum {stratum} in {split} cannot provide {requested} unique structures.")
        for item in pool[: int(requested)]:
            selected.append({**item, "source_split": split, "occupancy_stratum": int(stratum)})
    return selected


def build_field_eigen_selection_manifest(
    *,
    rho: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
    split_indices: Mapping[str, Sequence[int]],
    k_spec: KPathSpec,
    dataset_path: str,
    dataset_sha256: str,
    contract_id: str,
    contract_version: int,
    split_counts: Mapping[str, int],
    n_occupancy_strata: int,
    n_k_strata: int,
    band_range_one_based: Sequence[int],
    selector_seed: int,
) -> dict[str, Any]:
    """Build an immutable, balanced `(structure, k-point, band)` selection."""

    values = np.asarray(rho)
    if values.ndim < 2:
        raise ValueError("rho must contain a sample axis and at least one structure axis.")
    if len(metadata_rows) != int(values.shape[0]):
        raise ValueError("metadata_rows must align exactly with rho sample rows.")
    expected_total = int(sum(int(value) for value in split_counts.values()))
    if expected_total <= 0:
        raise ValueError("The field selection must request at least one record.")
    selected: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        requested = int(split_counts.get(split, 0))
        if requested:
            if split not in split_indices:
                raise ValueError(f"Missing source split indices for {split}.")
            selected.extend(
                _select_split_structures(
                    rho=values,
                    metadata_rows=metadata_rows,
                    source_indices=split_indices[split],
                    count=requested,
                    n_strata=int(n_occupancy_strata),
                    split=split,
                    seed=int(selector_seed),
                )
            )
    if len({row["source_sample_index"] for row in selected}) != expected_total:
        raise RuntimeError("Field selection must use distinct source structures.")

    physical_ids = physical_k_ids(k_spec.k_points)
    physical_representatives = [index for index, value in enumerate(physical_ids) if value not in physical_ids[:index]]
    if int(n_k_strata) > len(physical_representatives):
        raise ValueError("Requested more k strata than unique physical k-points.")
    positions = np.linspace(0, len(physical_representatives) - 1, int(n_k_strata), dtype=np.int64)
    k_indices = [int(physical_representatives[int(position)]) for position in positions]
    band_start, band_end = (int(item) for item in band_range_one_based)
    bands = list(range(band_start, band_end + 1))
    if not bands:
        raise ValueError("band_range_one_based must contain at least one band.")

    rng = np.random.default_rng(int(selector_seed))
    k_schedule = np.resize(np.asarray(k_indices, dtype=np.int64), expected_total)
    band_schedule = np.resize(np.asarray(bands, dtype=np.int64), expected_total)
    rng.shuffle(k_schedule)
    rng.shuffle(band_schedule)
    records = []
    for ordinal, source in enumerate(selected):
        k_index = int(k_schedule[ordinal])
        record = {
            "record_ordinal": int(ordinal),
            **{key: value for key, value in source.items() if key != "stable_key"},
            "k_index": k_index,
            "k_point": [float(value) for value in k_spec.k_points[k_index]],
            "physical_k_id": str(physical_ids[k_index]),
            "k_stratum": int(k_indices.index(k_index)),
            "band_index_one_based": int(band_schedule[ordinal]),
        }
        records.append(record)
    payload: dict[str, Any] = {
        "schema_version": "phc2d.field_eigen_selection.v1",
        "selection_id": f"field_eigen_te_{expected_total}_v1",
        "contract_id": str(contract_id),
        "contract_version": int(contract_version),
        "dataset_path": str(dataset_path),
        "dataset_sha256": str(dataset_sha256),
        "k_path_id": k_spec.k_path_id,
        "k_points_hash": k_spec.k_points_hash,
        "selector": {
            "seed": int(selector_seed),
            "source_split_counts": {name: int(split_counts.get(name, 0)) for name in ("train", "val", "test")},
            "occupancy_strata": int(n_occupancy_strata),
            "k_strata": int(n_k_strata),
            "band_range_one_based": [band_start, band_end],
            "band_count_policy": "balanced_with_max_difference_one",
        },
        "records": records,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    return payload


def build_field_eigen_operator_selection_manifest(
    *,
    rho: np.ndarray,
    metadata_rows: Sequence[Mapping[str, Any]],
    split_indices: Mapping[str, Sequence[int]],
    k_spec: KPathSpec,
    dataset_path: str,
    dataset_sha256: str,
    contract_id: str,
    contract_version: int,
    structure_split_counts: Mapping[str, int],
    n_occupancy_strata: int,
    n_k_strata: int,
    k_per_structure: int,
    band_range_one_based: Sequence[int],
    selector_seed: int,
) -> dict[str, Any]:
    """Build a deterministic repeated-structure operator-learning manifest.

    Each selected source structure is paired with a balanced subset of the
    physical-k strata and every requested band.  Records remain the atomic
    serialized unit while ``solver_job_id`` groups records that share one MPB
    solve.
    """

    values = np.asarray(rho)
    if values.ndim < 2 or len(metadata_rows) != int(values.shape[0]):
        raise ValueError("rho and metadata_rows must expose the same sample axis.")
    if int(k_per_structure) <= 0 or int(k_per_structure) > int(n_k_strata):
        raise ValueError("k_per_structure must be within 1..n_k_strata.")
    selected: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        count = int(structure_split_counts.get(split, 0))
        if count:
            selected.extend(
                _select_split_structures(
                    rho=values,
                    metadata_rows=metadata_rows,
                    source_indices=split_indices[split],
                    count=count,
                    n_strata=int(n_occupancy_strata),
                    split=split,
                    seed=int(selector_seed),
                )
            )
    expected_structures = sum(int(value) for value in structure_split_counts.values())
    if len(selected) != expected_structures or len({row["source_sample_index"] for row in selected}) != expected_structures:
        raise RuntimeError("Operator field selection must use the requested number of unique structures.")

    physical_ids = physical_k_ids(k_spec.k_points)
    representatives = [index for index, value in enumerate(physical_ids) if value not in physical_ids[:index]]
    if int(n_k_strata) > len(representatives):
        raise ValueError("Requested more k strata than unique physical k-points.")
    positions = np.linspace(0, len(representatives) - 1, int(n_k_strata), dtype=np.int64)
    k_indices = [int(representatives[int(position)]) for position in positions]
    band_start, band_end = (int(value) for value in band_range_one_based)
    bands = list(range(band_start, band_end + 1))
    if not bands:
        raise ValueError("band_range_one_based must contain at least one band.")

    records: list[dict[str, Any]] = []
    solver_jobs: list[dict[str, Any]] = []
    ordinal = 0
    for split in ("train", "val", "test"):
        split_rows = [row for row in selected if row["source_split"] == split]
        split_rows.sort(key=lambda row: (row["occupancy_stratum"], row["stable_key"]))
        for structure_ordinal, source in enumerate(split_rows):
            # A cyclic difference set gives every stratum equal global use when
            # the split structure count is divisible by n_k_strata.
            offsets = [0, 1, 3, 4, 6, 7, 2, 5][: int(k_per_structure)]
            strata = [int((structure_ordinal + offset) % int(n_k_strata)) for offset in offsets]
            for k_stratum in strata:
                k_index = int(k_indices[k_stratum])
                job_id = f"s{int(source['source_sample_index']):06d}_k{k_index:04d}"
                solver_jobs.append(
                    {
                        "solver_job_id": job_id,
                        "source_sample_index": int(source["source_sample_index"]),
                        "source_split": split,
                        "k_index": k_index,
                        "k_stratum": k_stratum,
                        "band_range_one_based": [band_start, band_end],
                    }
                )
                for band in bands:
                    records.append(
                        {
                            "record_ordinal": ordinal,
                            **{key: value for key, value in source.items() if key != "stable_key"},
                            "solver_job_id": job_id,
                            "k_index": k_index,
                            "k_point": [float(value) for value in k_spec.k_points[k_index]],
                            "physical_k_id": str(physical_ids[k_index]),
                            "k_stratum": k_stratum,
                            "band_index_one_based": int(band),
                        }
                    )
                    ordinal += 1

    record_split_counts = {
        split: int(structure_split_counts.get(split, 0)) * int(k_per_structure) * len(bands)
        for split in ("train", "val", "test")
    }
    payload: dict[str, Any] = {
        "schema_version": "phc2d.field_eigen_selection.v2",
        "selection_id": f"field_eigen_te_operator_{len(records)}_v1",
        "contract_id": str(contract_id),
        "contract_version": int(contract_version),
        "dataset_path": str(dataset_path),
        "dataset_sha256": str(dataset_sha256),
        "k_path_id": k_spec.k_path_id,
        "k_points_hash": k_spec.k_points_hash,
        "selector": {
            "seed": int(selector_seed),
            "source_structure_split_counts": {name: int(structure_split_counts.get(name, 0)) for name in ("train", "val", "test")},
            "source_split_counts": record_split_counts,
            "occupancy_strata": int(n_occupancy_strata),
            "k_strata": int(n_k_strata),
            "k_per_structure": int(k_per_structure),
            "band_range_one_based": [band_start, band_end],
            "band_count_policy": "all_bands_per_structure_k",
            "solver_shard_unit": "structure_k_all_bands",
        },
        "solver_jobs": solver_jobs,
        "records": records,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    return payload


def validate_field_eigen_selection_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return validation problems for a frozen field-eigen selection manifest."""

    problems: list[str] = []
    records = list(manifest.get("records") or [])
    selector = dict(manifest.get("selector") or {})
    expected_total = int(sum(int(value) for value in dict(selector.get("source_split_counts") or {}).values()))
    if len(records) != expected_total:
        problems.append(f"record count {len(records)} does not match requested split total {expected_total}")
    indices = [int(row.get("source_sample_index", -1)) for row in records]
    schema_version = str(manifest.get("schema_version", ""))
    if schema_version.endswith(".v1") and len(indices) != len(set(indices)):
        problems.append("source structures are not unique")
    if schema_version.endswith(".v2"):
        structure_counts = dict(selector.get("source_structure_split_counts") or {})
        observed_structures = {
            split: len({int(row.get("source_sample_index", -1)) for row in records if str(row.get("source_split", "")) == split})
            for split in ("train", "val", "test")
        }
        for split, expected in structure_counts.items():
            if observed_structures[str(split)] != int(expected):
                problems.append(f"split {split} has {observed_structures[str(split)]} structures, expected {expected}")
        tuples = [(int(row.get("source_sample_index", -1)), int(row.get("k_index", -1)), int(row.get("band_index_one_based", -1))) for row in records]
        if len(tuples) != len(set(tuples)):
            problems.append("operator records are not unique by structure/k/band")
        k_per_structure = int(selector.get("k_per_structure", 0))
        band_start, band_end = (int(value) for value in selector.get("band_range_one_based", (0, -1)))
        expected_per_structure = k_per_structure * max(0, band_end - band_start + 1)
        per_structure = Counter(indices)
        if per_structure and set(per_structure.values()) != {expected_per_structure}:
            problems.append("operator records per structure do not match k_per_structure times band count")
    observed_splits = Counter(str(row.get("source_split", "")) for row in records)
    for name, expected in dict(selector.get("source_split_counts") or {}).items():
        if observed_splits[str(name)] != int(expected):
            problems.append(f"split {name} has {observed_splits[str(name)]}, expected {expected}")
    bands = [int(row.get("band_index_one_based", -1)) for row in records]
    if bands and max(Counter(bands).values()) - min(Counter(bands).values()) > 1:
        problems.append("band allocation differs by more than one record")
    copy = dict(manifest)
    supplied_hash = str(copy.pop("manifest_sha256", ""))
    if supplied_hash != _canonical_hash(copy):
        problems.append("manifest_sha256 does not match its canonical payload")
    return problems


__all__ = [
    "build_field_eigen_operator_selection_manifest",
    "build_field_eigen_selection_manifest",
    "validate_field_eigen_selection_manifest",
]
