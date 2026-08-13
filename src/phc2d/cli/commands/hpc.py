"""HPC campaign command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    hpc = sub.add_parser(
        "hpc",
        help="Portable plan, shard, finalize, status, and audit controls for physics campaigns.",
    )
    hpc_sub = hpc.add_subparsers(dest="action", required=True)
    hpc_adapters = hpc_sub.add_parser("adapters", help="List registered physical task adapters.")
    hpc_adapters.set_defaults(command="hpc adapters")
    hpc_plan = hpc_sub.add_parser("plan", help="Freeze a deterministic physical case plan.")
    hpc_plan.add_argument("--adapter", required=True)
    hpc_plan.add_argument("--config", type=Path, required=True)
    hpc_plan.add_argument("--run-dir", type=Path, required=True)
    hpc_plan.add_argument("--workers", default="auto")
    hpc_plan.add_argument("--shards", type=int, required=True)
    hpc_plan.add_argument("--platform", required=True)
    hpc_plan.set_defaults(command="hpc plan")
    hpc_benchmark = hpc_sub.add_parser(
        "benchmark", help="Select an admitted 1/2/4/8-node tier from timing evidence."
    )
    hpc_benchmark.add_argument("--platform-profile", type=Path, required=True)
    benchmark_source = hpc_benchmark.add_mutually_exclusive_group(required=True)
    benchmark_source.add_argument("--evidence", type=Path)
    benchmark_source.add_argument("--partition", type=Path)
    hpc_benchmark.add_argument("--adapter")
    hpc_benchmark.add_argument("--solver-fingerprint")
    hpc_benchmark.add_argument("--environment-fingerprint")
    hpc_benchmark.add_argument("--admitted-tiers", nargs="+", type=int, default=[1])
    hpc_benchmark.add_argument("--output-evidence", type=Path)
    hpc_benchmark.add_argument("--case-count", type=int, required=True)
    hpc_benchmark.add_argument("--target-hours", type=float, default=3.0)
    hpc_benchmark.set_defaults(command="hpc benchmark")
    hpc_solve = hpc_sub.add_parser("solve-shard", help="Solve one exclusive physical shard.")
    hpc_solve.add_argument("--adapter", required=True)
    hpc_solve.add_argument("--config", type=Path, required=True)
    hpc_solve.add_argument("--run-dir", type=Path, required=True)
    hpc_solve.add_argument("--shard-index", type=int, required=True)
    hpc_solve.add_argument("--shard-count", type=int, required=True)
    hpc_solve.add_argument("--workers", default="auto")
    hpc_solve.add_argument("--allow-resume", action="store_true")
    hpc_solve.add_argument("--skip-environment-check", action="store_true")
    hpc_solve.set_defaults(command="hpc solve-shard")
    hpc_finalize = hpc_sub.add_parser(
        "finalize", help="Validate all shards and atomically publish a canonical dataset."
    )
    hpc_finalize.add_argument("--adapter", required=True)
    hpc_finalize.add_argument("--config", type=Path, required=True)
    hpc_finalize.add_argument("--run-dir", type=Path, required=True)
    hpc_finalize.add_argument("--workers", default="auto")
    hpc_finalize.set_defaults(command="hpc finalize")
    hpc_status = hpc_sub.add_parser("status", help="Read fixed-size campaign progress files.")
    hpc_status.add_argument("--run-dir", type=Path, required=True)
    hpc_status.add_argument("--write", action="store_true")
    hpc_status.set_defaults(command="hpc status")
    hpc_audit = hpc_sub.add_parser(
        "audit", help="Audit final artifacts without promoting acceptance."
    )
    hpc_audit.add_argument("--run-dir", type=Path, required=True)
    hpc_audit.add_argument("--skip-dataset-hash", action="store_true")
    hpc_audit.set_defaults(command="hpc audit")


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command == "hpc adapters":
        from phc2d.hpc import list_adapter_specs

        print(
            json.dumps(
                [item.to_mapping() for item in list_adapter_specs()],
                ensure_ascii=False,
                indent=2,
            )
        )
        return True
    if ns.command == "hpc plan":
        from phc2d.hpc import get_runtime_adapter

        adapter = get_runtime_adapter(ns.adapter)
        workers: int | str = ns.workers
        if str(workers).lower() != "auto":
            workers = int(workers)
        print(
            adapter.plan(
                ns.config,
                run_dir=ns.run_dir,
                workers=workers,
                shard_count=ns.shards,
                platform_id=ns.platform,
            )
        )
        return True
    if ns.command == "hpc benchmark":
        from phc2d.hpc import (
            BenchmarkEvidence,
            atomic_write_json,
            benchmark_evidence_from_partition,
            load_platform_profile,
            recommend_node_count,
        )

        if ns.evidence is not None:
            evidence = BenchmarkEvidence.from_mapping(
                json.loads(ns.evidence.read_text(encoding="utf-8"))
            )
        else:
            if not ns.adapter or not ns.solver_fingerprint or not ns.environment_fingerprint:
                raise SystemExit(
                    "--partition requires --adapter, --solver-fingerprint, and "
                    "--environment-fingerprint."
                )
            evidence = benchmark_evidence_from_partition(
                ns.partition,
                adapter_id=ns.adapter,
                solver_fingerprint=ns.solver_fingerprint,
                environment_fingerprint=ns.environment_fingerprint,
                admitted_node_tiers=ns.admitted_tiers,
            )
            if ns.output_evidence is not None:
                atomic_write_json(ns.output_evidence, evidence.to_mapping())
        result = recommend_node_count(
            case_count=ns.case_count,
            median_case_time_s=evidence.median_case_time_s,
            platform=load_platform_profile(ns.platform_profile),
            target_wall_time_s=ns.target_hours * 60.0 * 60.0,
            admitted_tiers=evidence.admitted_node_tiers,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return True
    if ns.command == "hpc solve-shard":
        from phc2d.hpc import get_runtime_adapter

        adapter = get_runtime_adapter(ns.adapter)
        workers = ns.workers if str(ns.workers).lower() == "auto" else int(ns.workers)
        print(
            adapter.solve_shard(
                ns.config,
                run_dir=ns.run_dir,
                shard_index=ns.shard_index,
                shard_count=ns.shard_count,
                workers=workers,
                allow_resume=ns.allow_resume,
                skip_environment_check=ns.skip_environment_check,
            )
        )
        return True
    if ns.command == "hpc finalize":
        from phc2d.hpc import get_runtime_adapter

        adapter = get_runtime_adapter(ns.adapter)
        workers = ns.workers if str(ns.workers).lower() == "auto" else int(ns.workers)
        print(adapter.finalize(ns.config, run_dir=ns.run_dir, workers=workers))
        return True
    if ns.command == "hpc status":
        from phc2d.hpc import collect_campaign_status

        print(
            json.dumps(
                collect_campaign_status(ns.run_dir, write=ns.write), ensure_ascii=False, indent=2
            )
        )
        return True
    if ns.command == "hpc audit":
        from phc2d.hpc import audit_campaign

        payload = audit_campaign(ns.run_dir, verify_dataset_hash=not ns.skip_dataset_hash)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if payload["status"] != "passed":
            raise SystemExit(2)
        return True
    return False
