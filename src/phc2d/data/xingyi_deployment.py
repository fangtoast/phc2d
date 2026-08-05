"""Auditable worker selection for Xingyi full-cell benchmarks.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from phc2d.runs import write_json


def select_xingyi_worker_count(
    rows: Sequence[dict[str, Any]],
    *,
    memory_limit_gib: float = 400.0,
    near_tie_fraction: float = 0.10,
) -> dict[str, Any]:
    """Choose the lowest worker count within 10% of the fastest valid run."""

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
        "status": "xingyi_worker_selection_pass",
        "selected_workers": int(selected["workers"]),
        "fastest_workers": int(fastest["workers"]),
        "fastest_throughput_samples_per_s": float(fastest["throughput_samples_per_s"]),
        "near_tie_threshold_samples_per_s": threshold,
        "near_tie_fraction": float(near_tie_fraction),
        "memory_limit_gib": float(memory_limit_gib),
        "valid_worker_counts": sorted(int(row["workers"]) for row in valid),
        "benchmarks": sorted((dict(row) for row in rows), key=lambda row: int(row["workers"])),
    }


def collect_xingyi_benchmarks(root: str | Path) -> list[dict[str, Any]]:
    """Load benchmark metrics and process-tree peak memory from one benchmark root."""

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(Path(root).glob("full_cell_binary_xingyi_benchmark64_w*_*")):
        metrics_path = run_dir / "metrics.json"
        results_path = run_dir / "results.json"
        memory_path = run_dir / "peak_process_tree_rss_bytes.txt"
        if not metrics_path.is_file() or not results_path.is_file() or not memory_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results = json.loads(results_path.read_text(encoding="utf-8"))
        worker_token = run_dir.name.split("_w", 1)[1].split("_", 1)[0]
        peak_bytes = int(memory_path.read_text(encoding="utf-8").strip())
        rows.append(
            {
                "workers": int(worker_token),
                "run_dir": str(run_dir),
                "throughput_samples_per_s": float(metrics.get("worker_throughput_samples_per_s", 0.0)),
                "failed_count": int(metrics.get("failed_count", results.get("failure_count", -1))),
                "peak_rss_gib": peak_bytes / float(1024**3),
                "oom": "oom" in str(results.get("status", "")).lower(),
                "status": str(results.get("status", "")),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Select and persist the production worker count from benchmark artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    decision = select_xingyi_worker_count(collect_xingyi_benchmarks(args.benchmark_root))
    write_json(args.output, decision)
    print(decision["selected_workers"])


if __name__ == "__main__":
    main()


__all__ = ["collect_xingyi_benchmarks", "select_xingyi_worker_count"]
