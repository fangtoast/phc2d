"""Evidence-based resource selection for portable Slurm physics campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class PlatformProfile:
    """Scheduler limits and one-node resources for a supported cluster."""

    platform_id: str
    scheduler: str
    partition: str
    cores_per_node: int
    memory_gb_per_node: int
    max_nodes: int
    max_array_tasks: int
    allowed_node_tiers: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self) -> None:
        if self.scheduler != "slurm":
            raise ValueError("Only Slurm platform profiles are supported")
        if min(self.cores_per_node, self.memory_gb_per_node, self.max_nodes, self.max_array_tasks) <= 0:
            raise ValueError("Platform numeric resources must be positive")
        if not self.allowed_node_tiers or tuple(sorted(set(self.allowed_node_tiers))) != self.allowed_node_tiers:
            raise ValueError("allowed_node_tiers must be unique and ascending")
        if self.allowed_node_tiers[-1] > self.max_nodes:
            raise ValueError("allowed_node_tiers cannot exceed max_nodes")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlatformProfile":
        payload = dict(value)
        return cls(
            platform_id=str(payload["platform_id"]),
            scheduler=str(payload.get("scheduler", "slurm")),
            partition=str(payload["partition"]),
            cores_per_node=int(payload["cores_per_node"]),
            memory_gb_per_node=int(payload["memory_gb_per_node"]),
            max_nodes=int(payload["max_nodes"]),
            max_array_tasks=int(payload["max_array_tasks"]),
            allowed_node_tiers=tuple(int(item) for item in payload.get("allowed_node_tiers", [1, 2, 4, 8])),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_node_tiers"] = list(self.allowed_node_tiers)
        return payload


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Reusable per-case timing evidence for one solver fingerprint."""

    adapter_id: str
    solver_fingerprint: str
    environment_fingerprint: str
    sample_count: int
    median_case_time_s: float
    p95_case_time_s: float
    peak_memory_mb: float
    median_output_bytes: float
    admitted_node_tiers: tuple[int, ...] = (1,)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkEvidence":
        payload = dict(value)
        result = cls(
            adapter_id=str(payload["adapter_id"]),
            solver_fingerprint=str(payload["solver_fingerprint"]),
            environment_fingerprint=str(payload["environment_fingerprint"]),
            sample_count=int(payload["sample_count"]),
            median_case_time_s=float(payload["median_case_time_s"]),
            p95_case_time_s=float(payload["p95_case_time_s"]),
            peak_memory_mb=float(payload.get("peak_memory_mb", 0.0)),
            median_output_bytes=float(payload.get("median_output_bytes", 0.0)),
            admitted_node_tiers=tuple(
                int(item) for item in payload.get("admitted_node_tiers", [1])
            ),
        )
        if result.sample_count <= 0 or result.median_case_time_s <= 0.0:
            raise ValueError("Benchmark sample count and median time must be positive")
        if not result.admitted_node_tiers or result.admitted_node_tiers[0] != 1:
            raise ValueError("Benchmark evidence must admit the one-node tier")
        return result

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["admitted_node_tiers"] = list(self.admitted_node_tiers)
        return payload


def benchmark_evidence_from_partition(
    partition_path: str | Path,
    *,
    adapter_id: str,
    solver_fingerprint: str,
    environment_fingerprint: str,
    admitted_node_tiers: Sequence[int] = (1,),
) -> BenchmarkEvidence:
    """Summarize successful case timings from one benchmark partition."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required to summarize partition benchmarks") from exc
    import numpy as np

    path = Path(partition_path)
    with h5py.File(path, "r") as handle:
        success = np.asarray(handle["success"], dtype=np.uint8).astype(bool)
        times = np.asarray(handle["solve_time_s"], dtype=np.float64)[success]
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("Benchmark partition has no positive finite successful solve times")
    return BenchmarkEvidence(
        adapter_id=adapter_id,
        solver_fingerprint=solver_fingerprint,
        environment_fingerprint=environment_fingerprint,
        sample_count=int(times.size),
        median_case_time_s=float(np.median(times)),
        p95_case_time_s=float(np.percentile(times, 95.0)),
        peak_memory_mb=0.0,
        median_output_bytes=float(os.path.getsize(path) / max(int(times.size), 1)),
        admitted_node_tiers=tuple(int(item) for item in admitted_node_tiers),
    )


def load_platform_profile(path: str | Path) -> PlatformProfile:
    """Load a platform profile without importing solver dependencies."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Platform profile must be a YAML mapping")
    return PlatformProfile.from_mapping(payload)


def recommend_node_count(
    *,
    case_count: int,
    median_case_time_s: float,
    platform: PlatformProfile,
    target_wall_time_s: float = 3.0 * 60.0 * 60.0,
    parallel_efficiency: float = 0.75,
    admitted_tiers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Choose the smallest admitted node tier that meets the wall-time target."""

    total = int(case_count)
    median = float(median_case_time_s)
    target = float(target_wall_time_s)
    efficiency = float(parallel_efficiency)
    if total <= 0 or median <= 0.0 or target <= 0.0:
        raise ValueError("case_count, median_case_time_s, and target_wall_time_s must be positive")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("parallel_efficiency must be in (0, 1]")
    tiers = tuple(int(item) for item in (admitted_tiers or platform.allowed_node_tiers))
    tiers = tuple(item for item in tiers if item in platform.allowed_node_tiers)
    if not tiers:
        raise ValueError("No admitted resource tier is supported by the platform")
    raw_nodes = math.ceil(total * median / (platform.cores_per_node * efficiency * target))
    selected = next((tier for tier in tiers if tier >= raw_nodes), tiers[-1])
    estimate = total * median / (platform.cores_per_node * efficiency * selected)
    target_met = bool(estimate <= target)
    return {
        "selected_nodes": int(selected),
        "workers_per_node": int(platform.cores_per_node),
        "total_workers": int(selected * platform.cores_per_node),
        "estimated_wall_time_s": float(estimate),
        "target_wall_time_s": target,
        "parallel_efficiency_assumption": efficiency,
        "target_met": target_met,
        "resource_limited": bool(not target_met and selected == tiers[-1]),
        "allowed_node_tiers": list(tiers),
    }


def admitted_scaling_tiers(
    measurements: Mapping[int, float],
    *,
    four_node_min_speedup: float = 3.0,
    eight_node_min_incremental_efficiency: float = 0.5,
) -> tuple[int, ...]:
    """Return tiers admitted by measured aggregate throughput."""

    throughput = {int(nodes): float(value) for nodes, value in measurements.items()}
    if 1 not in throughput or throughput[1] <= 0.0:
        raise ValueError("One-node throughput evidence is required")
    admitted = [1]
    if 2 in throughput and throughput[2] >= throughput[1] * 1.5:
        admitted.append(2)
    if 4 in throughput and throughput[4] >= throughput[1] * float(four_node_min_speedup):
        admitted.append(4)
    if 8 in throughput and 4 in admitted:
        incremental = throughput[8] / max(throughput[4] * 2.0, 1.0e-12)
        if incremental >= float(eight_node_min_incremental_efficiency):
            admitted.append(8)
    return tuple(admitted)


__all__ = [
    "BenchmarkEvidence",
    "PlatformProfile",
    "admitted_scaling_tiers",
    "benchmark_evidence_from_partition",
    "load_platform_profile",
    "recommend_node_count",
]
