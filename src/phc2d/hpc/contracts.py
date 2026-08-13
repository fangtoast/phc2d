"""Portable contracts for evidence-gated HPC physics campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


CAMPAIGN_STATES = (
    "plan_ready",
    "solver_running",
    "solver_shards_complete",
    "finalizing",
    "cluster_generated_pending_download",
    "downloaded_hash_verified",
    "cluster_generated_pending_wsl_acceptance",
    "cross_environment_accepted",
)


def canonical_json(value: Any) -> str:
    """Return stable UTF-8 JSON used by all campaign fingerprints."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_payload(value: Any) -> str:
    """Hash a JSON-compatible value with canonical serialization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a small JSON control-plane artifact."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, out)
    return out


@dataclass(frozen=True)
class PhysicalCase:
    """One deterministic, independently solvable physics case."""

    case_id: str
    global_index: int
    payload: Mapping[str, Any]
    split: str = ""
    seed: int | None = None
    input_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if int(self.global_index) < 0:
            raise ValueError("global_index must be non-negative")
        if self.input_sha256 and len(self.input_sha256) != 64:
            raise ValueError("input_sha256 must be a SHA-256 hex digest")

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["input_sha256"]:
            payload["input_sha256"] = sha256_payload(
                {
                    "case_id": self.case_id,
                    "global_index": int(self.global_index),
                    "payload": dict(self.payload),
                    "split": self.split,
                    "seed": self.seed,
                }
            )
        return payload


@dataclass(frozen=True)
class PhysicalPlanManifest:
    """Immutable identity and output contract for a physical case plan."""

    adapter_id: str
    case_count: int
    plan_sha256: str
    solver_fingerprint: str
    output_schema: str
    case_plan: str = "physical_cases.jsonl"
    schema_version: str = "phc2d.physical_plan.v1"

    def __post_init__(self) -> None:
        if int(self.case_count) <= 0:
            raise ValueError("case_count must be positive")
        if len(self.plan_sha256) != 64:
            raise ValueError("plan_sha256 must be a SHA-256 hex digest")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalShardSpec:
    """One exclusive contiguous slice of an immutable physical plan."""

    shard_index: int
    shard_count: int
    case_start: int
    case_end: int
    workers: int
    plan_sha256: str
    schema_version: str = "phc2d.physical_shard.v1"

    def __post_init__(self) -> None:
        if not 0 <= int(self.shard_index) < int(self.shard_count):
            raise ValueError("shard_index must be in [0, shard_count)")
        if int(self.case_start) < 0 or int(self.case_end) <= int(self.case_start):
            raise ValueError("shard case range must be non-empty and non-negative")
        if int(self.workers) <= 0:
            raise ValueError("workers must be positive")
        if len(self.plan_sha256) != 64:
            raise ValueError("plan_sha256 must be a SHA-256 hex digest")

    @classmethod
    def partition(
        cls,
        *,
        case_count: int,
        shard_index: int,
        shard_count: int,
        workers: int,
        plan_sha256: str,
    ) -> "PhysicalShardSpec":
        """Partition cases into balanced, stable contiguous ranges."""

        total = int(case_count)
        parts = int(shard_count)
        index = int(shard_index)
        if total <= 0 or parts <= 0 or parts > total:
            raise ValueError("shard_count must be positive and no greater than case_count")
        if not 0 <= index < parts:
            raise ValueError("shard_index must be in [0, shard_count)")
        base, remainder = divmod(total, parts)
        start = index * base + min(index, remainder)
        end = start + base + (1 if index < remainder else 0)
        return cls(index, parts, start, end, int(workers), plan_sha256)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalResultRecord:
    """Case-level status and provenance independent of the result tensor type."""

    case_id: str
    global_index: int
    status: str
    solve_time_s: float
    worker_id: str
    input_sha256: str
    output_sha256: str = ""
    error: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalCampaignManifest:
    """Small control-plane summary for a multi-stage physical campaign."""

    campaign_id: str
    adapter_id: str
    state: str
    plan_sha256: str
    planned_cases: int
    shard_count: int
    platform_id: str
    job_ids: Mapping[str, Any] = field(default_factory=dict)
    next_gate: str = ""
    schema_version: str = "phc2d.physical_campaign.v1"

    def __post_init__(self) -> None:
        if self.state not in CAMPAIGN_STATES:
            raise ValueError(f"Unknown campaign state: {self.state}")
        if int(self.planned_cases) <= 0 or int(self.shard_count) <= 0:
            raise ValueError("planned_cases and shard_count must be positive")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def validate_shard_coverage(specs: Sequence[PhysicalShardSpec], *, case_count: int) -> None:
    """Fail unless shard ranges cover every case exactly once."""

    ordered = sorted(specs, key=lambda item: item.case_start)
    if not ordered:
        raise ValueError("At least one shard is required")
    cursor = 0
    for expected_index, spec in enumerate(ordered):
        if spec.shard_index != expected_index:
            raise ValueError("Shard indices must be unique and contiguous")
        if spec.case_start != cursor:
            raise ValueError("Shard ranges contain a gap or overlap")
        cursor = spec.case_end
    if cursor != int(case_count):
        raise ValueError("Shard ranges do not cover the complete plan")


__all__ = [
    "CAMPAIGN_STATES",
    "PhysicalCampaignManifest",
    "PhysicalCase",
    "PhysicalPlanManifest",
    "PhysicalResultRecord",
    "PhysicalShardSpec",
    "atomic_write_json",
    "canonical_json",
    "sha256_file",
    "sha256_payload",
    "validate_shard_coverage",
]
