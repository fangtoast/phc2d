"""Launch independent Query Dataset V2 shard processes and merge them.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.run_query_dataset_v2_pilot_generation import ROOT, _jsonable, _write_json, load_query_pilot_config


@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    start: int
    end: int

    @property
    def name(self) -> str:
        return f"shard_{self.shard_index:03d}"


def build_shard_specs(total_samples: int, shard_size: int) -> list[ShardSpec]:
    """Split ``total_samples`` into contiguous shard ranges."""

    total = int(total_samples)
    size = int(shard_size)
    if total <= 0:
        raise ValueError("total_samples must be positive")
    if size <= 0:
        raise ValueError("shard_size must be positive")
    specs: list[ShardSpec] = []
    for shard_index, start in enumerate(range(0, total, size)):
        specs.append(ShardSpec(shard_index=shard_index, start=start, end=min(start + size, total)))
    return specs


def _python_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join(["src", "."])
    env["PYTHONPATH"] = prefix if not existing else os.pathsep.join([prefix, existing])
    return env


def _launch_shard(
    spec: ShardSpec,
    *,
    config_path: Path,
    run_root: Path,
    workers_per_shard: int,
    worker_backend: str,
    python_executable: str,
) -> subprocess.Popen:
    shard_path = run_root / "shards" / f"{spec.name}.npz"
    shard_run_dir = run_root / "shard_runs" / spec.name
    shard_run_dir.mkdir(parents=True, exist_ok=True)
    stdout = (shard_run_dir / "launcher.out").open("w", encoding="utf-8")
    stderr = (shard_run_dir / "launcher.err").open("w", encoding="utf-8")
    command = [
        python_executable,
        "scripts/generate_query_dataset_v2_shard.py",
        "--config",
        str(config_path),
        "--start",
        str(spec.start),
        "--end",
        str(spec.end),
        "--out",
        str(shard_path),
        "--run-dir",
        str(shard_run_dir),
        "--workers",
        str(workers_per_shard),
        "--worker-backend",
        str(worker_backend),
    ]
    return subprocess.Popen(command, cwd=ROOT, env=_python_env(), stdout=stdout, stderr=stderr, text=True)


def run_sharded_generation(
    config_path: str | Path,
    *,
    run_root: str | Path,
    shard_size: int = 250,
    parallel_jobs: int = 1,
    workers_per_shard: int = 1,
    worker_backend: str = "processpool",
    output_path: str | Path | None = None,
    merge: bool = True,
    python_executable: str | None = None,
) -> Path:
    """Run shard processes with bounded outer parallelism, then merge outputs."""

    started = time.perf_counter()
    config_path = Path(config_path)
    cfg = load_query_pilot_config(config_path)
    total_samples = int(dict(cfg.get("dataset") or {})["sample_count"])
    run_root = Path(run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "shards").mkdir(parents=True, exist_ok=True)
    (run_root / "shard_runs").mkdir(parents=True, exist_ok=True)
    specs = build_shard_specs(total_samples, shard_size)
    parallel_jobs = max(1, int(parallel_jobs))
    python_executable = python_executable or sys.executable

    state: dict[str, Any] = {
        "status": "running",
        "config_path": str(config_path),
        "run_root": str(run_root),
        "total_samples": total_samples,
        "shard_size": int(shard_size),
        "parallel_jobs": parallel_jobs,
        "workers_per_shard": int(workers_per_shard),
        "worker_backend": worker_backend,
        "shards": [{"name": spec.name, "start": spec.start, "end": spec.end, "status": "pending"} for spec in specs],
    }
    _write_json(run_root / "launcher_state.json", state)

    active: dict[int, tuple[ShardSpec, subprocess.Popen]] = {}
    pending = list(specs)
    completed: list[ShardSpec] = []

    def mark(spec: ShardSpec, status: str, **extra: Any) -> None:
        for row in state["shards"]:
            if row["name"] == spec.name:
                row.update({"status": status, **extra})
                break
        _write_json(run_root / "launcher_state.json", state)

    try:
        while pending or active:
            while pending and len(active) < parallel_jobs:
                spec = pending.pop(0)
                proc = _launch_shard(
                    spec,
                    config_path=config_path,
                    run_root=run_root,
                    workers_per_shard=workers_per_shard,
                    worker_backend=worker_backend,
                    python_executable=python_executable,
                )
                active[proc.pid] = (spec, proc)
                mark(spec, "running", pid=proc.pid)

            time.sleep(2.0)
            for pid, (spec, proc) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                active.pop(pid)
                if code != 0:
                    mark(spec, "failed", exit_code=code)
                    state["status"] = "failed"
                    _write_json(run_root / "launcher_state.json", state)
                    for _, other in active.values():
                        other.terminate()
                    raise RuntimeError(f"{spec.name} failed with exit code {code}")
                completed.append(spec)
                mark(spec, "completed", exit_code=code)
    finally:
        for _, proc in active.values():
            if proc.poll() is None:
                proc.terminate()

    merged_path = Path(output_path) if output_path is not None else Path(str(dict(cfg.get("dataset") or {})["output_path"]))
    if merge:
        merge_command = [
            python_executable,
            "scripts/merge_query_dataset_v2_shards.py",
            "--config",
            str(config_path),
            "--shard-dir",
            str(run_root / "shards"),
            "--pattern",
            "shard_*.npz",
            "--out",
            str(merged_path),
            "--run-dir",
            str(run_root / "merge"),
        ]
        subprocess.run(merge_command, cwd=ROOT, env=_python_env(), check=True)
    state["status"] = "completed"
    state["completed_shards"] = len(completed)
    state["wall_time_total_s"] = float(time.perf_counter() - started)
    state["merged_dataset_path"] = str(merged_path) if merge else ""
    _write_json(run_root / "launcher_state.json", state)
    _write_json(
        run_root / "results.json",
        {
            "action": "query_dataset_v2_sharded_generation",
            "status": "completed",
            "shard_count": len(specs),
            "parallel_jobs": parallel_jobs,
            "workers_per_shard": int(workers_per_shard),
            "merged_dataset_path": str(merged_path) if merge else "",
        },
    )
    _write_json(
        run_root / "metrics.json",
        {
            "physical_validation_decision": "shards_generated",
            "real_mpb_generation_completed": True,
            "wall_time_total_s": float(state["wall_time_total_s"]),
            "sample_count": total_samples,
            "shard_count": len(specs),
            "parallel_jobs": parallel_jobs,
            "workers_per_shard": int(workers_per_shard),
        },
    )
    return run_root


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run independent Query Dataset V2 MPB shards with outer parallelism.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--workers-per-shard", type=int, default=1)
    parser.add_argument("--worker-backend", default="processpool")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args(argv)
    return run_sharded_generation(
        args.config,
        run_root=args.run_root,
        shard_size=args.shard_size,
        parallel_jobs=args.parallel_jobs,
        workers_per_shard=args.workers_per_shard,
        worker_backend=args.worker_backend,
        output_path=args.out,
        merge=not args.no_merge,
    )


if __name__ == "__main__":
    main()
