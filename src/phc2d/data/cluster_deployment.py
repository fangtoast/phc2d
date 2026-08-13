"""Auditable preparation and worker selection for native-Linux clusters.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from phc2d.data.full_cell_generation import (
    _effective_split_counts,
    _load_frozen_sample_plan,
    _stable_stratified_splits,
    load_full_cell_generation_config,
)
from phc2d.runs import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_benchmark_sample_plan(
    source: str | Path,
    output: str | Path,
    config_path: str | Path,
    *,
    count: int,
) -> Path:
    """Derive a benchmark-only prefix while preserving frozen geometries."""

    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    cfg = load_full_cell_generation_config(config_path)
    full_count = int(dict(cfg["dataset"])["sample_count"])
    if int(count) < 1 or int(count) > full_count:
        raise ValueError(f"benchmark count must be in [1,{full_count}]")
    structures, strata, _ = _load_frozen_sample_plan(source_path, cfg, effective_count=full_count)
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    subset_structures = structures[: int(count)]
    subset_strata = strata[: int(count)]
    split_counts = _effective_split_counts(
        dict(dict(cfg["dataset"])["split"]),
        full_count,
        int(count),
    )
    _, _, _, labels = _stable_stratified_splits(
        [structure.sample_id for structure in subset_structures],
        subset_strata,
        split_counts,
    )
    derived: list[dict[str, Any]] = []
    for row, label in zip(rows[: int(count)], labels):
        derived.append({**row, "split": label})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in derived),
        encoding="utf-8",
    )
    write_json(
        output_path.with_suffix(output_path.suffix + ".manifest.json"),
        {
            "status": "benchmark_plan_derived",
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "output_path": str(output_path),
            "output_sha256": _sha256(output_path),
            "sample_count": int(count),
            "preserved_fields": ["sample_index", "sample_id", "generator_family", "stratum", "parameters", "geometry_hash"],
            "changed_fields": ["split"],
        },
    )
    return output_path


def collect_cluster_benchmarks(root: str | Path) -> list[dict[str, Any]]:
    """Load benchmark metrics and process-tree peak memory from a benchmark root."""

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in Path(root).iterdir() if path.is_dir() and "benchmark" in path.name):
        match = re.search(r"_w(\d+)(?:_|$)", run_dir.name)
        metrics_path = run_dir / "metrics.json"
        results_path = run_dir / "results.json"
        memory_path = run_dir / "peak_process_tree_rss_bytes.txt"
        if match is None or not metrics_path.is_file() or not results_path.is_file() or not memory_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        peak_bytes = int(memory_path.read_text(encoding="utf-8").strip())
        rows.append(
            {
                "workers": int(match.group(1)),
                "run_dir": str(run_dir),
                "throughput_samples_per_s": float(metrics.get("worker_throughput_samples_per_s", 0.0)),
                "failed_count": int(metrics.get("failed_count", results.get("failure_count", -1))),
                "peak_rss_gib": peak_bytes / float(1024**3),
                "oom": "oom" in str(results.get("status", "")).lower(),
                "status": str(results.get("status", "")),
            }
        )
    return rows


def should_run_extended_worker_benchmarks(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_workers: int = 96,
    candidate_workers: int = 128,
    minimum_gain_fraction: float = 0.10,
    memory_limit_gib: float = 500.0,
) -> bool:
    """Return whether the 192/256-worker extension is admissible."""

    by_workers = {int(row["workers"]): row for row in rows}
    if baseline_workers not in by_workers or candidate_workers not in by_workers:
        return False
    baseline = by_workers[baseline_workers]
    candidate = by_workers[candidate_workers]
    if any(int(row.get("failed_count", -1)) != 0 or bool(row.get("oom", False)) for row in (baseline, candidate)):
        return False
    baseline_throughput = float(baseline.get("throughput_samples_per_s", 0.0))
    candidate_throughput = float(candidate.get("throughput_samples_per_s", 0.0))
    return (
        baseline_throughput > 0.0
        and candidate_throughput >= baseline_throughput * (1.0 + float(minimum_gain_fraction))
        and float(candidate.get("peak_rss_gib", float("inf"))) < float(memory_limit_gib)
    )


def select_cluster_worker_count(
    rows: Sequence[dict[str, Any]],
    *,
    memory_limit_gib: float = 600.0,
    near_tie_fraction: float = 0.10,
) -> dict[str, Any]:
    """Choose the lowest worker count within ten percent of the fastest valid run."""

    valid = [
        dict(row)
        for row in rows
        if int(row.get("failed_count", -1)) == 0
        and not bool(row.get("oom", False))
        and float(row.get("peak_rss_gib", float("inf"))) < float(memory_limit_gib)
        and float(row.get("throughput_samples_per_s", 0.0)) > 0.0
    ]
    if not valid:
        raise ValueError("no worker benchmark passed zero-failure, no-OOM, memory, and throughput gates")
    fastest = max(valid, key=lambda row: float(row["throughput_samples_per_s"]))
    threshold = float(fastest["throughput_samples_per_s"]) * (1.0 - float(near_tie_fraction))
    near_tied = [row for row in valid if float(row["throughput_samples_per_s"]) >= threshold]
    selected = min(near_tied, key=lambda row: int(row["workers"]))
    return {
        "status": "cluster_worker_selection_pass",
        "selected_workers": int(selected["workers"]),
        "selected_peak_rss_gib": float(selected["peak_rss_gib"]),
        "selected_throughput_samples_per_s": float(selected["throughput_samples_per_s"]),
        "fastest_workers": int(fastest["workers"]),
        "fastest_throughput_samples_per_s": float(fastest["throughput_samples_per_s"]),
        "near_tie_threshold_samples_per_s": threshold,
        "near_tie_fraction": float(near_tie_fraction),
        "memory_limit_gib": float(memory_limit_gib),
        "valid_worker_counts": sorted(int(row["workers"]) for row in valid),
        "benchmarks": sorted((dict(row) for row in rows), key=lambda row: int(row["workers"])),
    }


def recommend_slurm_resources(
    decision: dict[str, Any],
    *,
    sample_count: int = 10000,
) -> dict[str, Any]:
    """Derive bounded memory and time requests from the selected benchmark."""

    memory_gib = min(700, max(64, int(math.ceil(float(decision["selected_peak_rss_gib"]) * 1.25 / 8.0) * 8)))
    raw_hours = 1.5 * int(sample_count) / float(decision["selected_throughput_samples_per_s"]) / 3600.0
    time_hours = max(24, int(math.ceil(raw_hours)))
    return {
        "cpus_per_task": int(decision["selected_workers"]),
        "memory_gib": memory_gib,
        "time_limit_hours": time_hours,
        "sample_count": int(sample_count),
        "time_factor": 1.5,
        "memory_factor": 1.25,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare benchmark plans and persist worker/resource decisions."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-plan")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--count", type=int, required=True)
    gate = subparsers.add_parser("extended-gate")
    gate.add_argument("--benchmark-root", type=Path, required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--benchmark-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--sample-count", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.command == "prepare-plan":
        print(prepare_benchmark_sample_plan(args.source, args.output, args.config, count=args.count))
        return
    rows = collect_cluster_benchmarks(args.benchmark_root)
    if args.command == "extended-gate":
        print("YES" if should_run_extended_worker_benchmarks(rows) else "NO")
        return
    decision = select_cluster_worker_count(rows)
    decision["slurm_resources"] = recommend_slurm_resources(decision, sample_count=args.sample_count)
    write_json(args.output, decision)
    print(decision["selected_workers"])


if __name__ == "__main__":
    main()


__all__ = [
    "collect_cluster_benchmarks",
    "prepare_benchmark_sample_plan",
    "recommend_slurm_resources",
    "select_cluster_worker_count",
    "should_run_extended_worker_benchmarks",
]
