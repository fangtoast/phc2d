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
from phc2d.data.full_cell_generation import (
    _effective_split_counts,
    _load_frozen_sample_plan,
    _stable_stratified_splits,
    _structure_schedule,
    load_full_cell_generation_config,
)


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


def prepare_xingyi_benchmark_plan(
    source: str | Path,
    output: str | Path,
    config_path: str | Path,
    *,
    count: int = 64,
) -> Path:
    """Write a validated frozen-geometry prefix with benchmark-local split labels."""

    cfg = load_full_cell_generation_config(config_path)
    source_rows = [
        json.loads(line)
        for line in Path(source).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requested = int(count)
    if requested <= 0 or len(source_rows) < requested:
        raise ValueError(f"benchmark plan requires at least {requested} source rows")
    rows = [dict(row) for row in source_rows[:requested]]
    experiment_id = str(dict(cfg["experiment"])["experiment_id"])
    lane = str(dict(cfg["dataset"])["lane"])
    schedule = _structure_schedule(lane, requested)
    sample_ids: list[str] = []
    strata: list[str] = []
    for index, (row, (family, _variant, stratum)) in enumerate(zip(rows, schedule)):
        expected_id = f"{experiment_id}_{index:06d}"
        if int(row.get("sample_index", -1)) != index or str(row.get("sample_id")) != expected_id:
            raise ValueError(f"benchmark source identity mismatch at row {index}")
        if str(row.get("generator_family")) != family or str(row.get("stratum")) != stratum:
            raise ValueError(f"benchmark source schedule mismatch at row {index}")
        sample_ids.append(expected_id)
        strata.append(stratum)
    dataset = dict(cfg["dataset"])
    split_counts = _effective_split_counts(
        dict(dataset["split"]),
        int(dataset["sample_count"]),
        requested,
    )
    _, _, _, labels = _stable_stratified_splits(sample_ids, strata, split_counts)
    for row, label in zip(rows, labels):
        row["split"] = label
        row["benchmark_split_reassigned"] = True
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    _load_frozen_sample_plan(out, cfg, effective_count=requested)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Select and persist the production worker count from benchmark artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-plan-source", type=Path)
    parser.add_argument("--prepare-plan-output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--count", type=int, default=64)
    args = parser.parse_args(argv)
    if args.prepare_plan_source is not None:
        if args.prepare_plan_output is None or args.config is None:
            parser.error("--prepare-plan-source requires --prepare-plan-output and --config")
        print(prepare_xingyi_benchmark_plan(args.prepare_plan_source, args.prepare_plan_output, args.config, count=args.count))
        return
    if args.benchmark_root is None or args.output is None:
        parser.error("worker selection requires --benchmark-root and --output")
    decision = select_xingyi_worker_count(collect_xingyi_benchmarks(args.benchmark_root))
    write_json(args.output, decision)
    print(decision["selected_workers"])


if __name__ == "__main__":
    main()


__all__ = ["collect_xingyi_benchmarks", "prepare_xingyi_benchmark_plan", "select_xingyi_worker_count"]
