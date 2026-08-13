"""Data command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phc2d.data import infer_dataset_schema


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    data = sub.add_parser("data", help="Dataset generation and data-schema utilities.")
    data_sub = data.add_subparsers(dest="action", required=True)
    generate = data_sub.add_parser(
        "generate",
        help="Generate an evidence-gated dataset from a research config.",
        description="Generate a full-cell dataset with real MPB, or run its solver preflight.",
    )
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--run-dir", type=Path, default=None)
    generate.add_argument("--max-samples", type=int, default=None)
    generate.add_argument(
        "--workers",
        type=str,
        default=None,
        help="Worker count or 'auto'; the config is used when omitted.",
    )
    generate.add_argument(
        "--plan-workers",
        type=str,
        default=None,
        help="Structure-plan process count or 'auto'; generator.plan_workers is used when omitted.",
    )
    generate.add_argument(
        "--skip-environment-check",
        action="store_true",
        help="Test-only escape hatch; never use this to claim accepted MPB validation.",
    )
    generate.set_defaults(command="data generate")
    prepare_plan = data_sub.add_parser(
        "prepare-full-cell-plan",
        help="Freeze a deterministic full-cell structure plan without invoking MPB.",
        description=(
            "Write and validate sample_plan.jsonl plus provenance before cluster generation."
        ),
    )
    prepare_plan.add_argument("--config", type=Path, required=True)
    prepare_plan.add_argument("--run-dir", type=Path, default=None)
    prepare_plan.add_argument(
        "--workers",
        type=str,
        default=None,
        help="Structure-plan process count or 'auto'; generator.plan_workers is used when omitted.",
    )
    prepare_plan.set_defaults(command="data prepare-full-cell-plan")
    accept_cluster = data_sub.add_parser(
        "accept-cluster",
        help="Recompute fixed cluster structures in WSL and gate a full-cell dataset.",
        description="Compare native-Linux MPB labels against WSL Ubuntu conda meep_env replay.",
    )
    accept_cluster.add_argument("--dataset", type=Path, required=True)
    accept_cluster.add_argument("--config", type=Path, required=True)
    accept_cluster.add_argument("--run-dir", type=Path, default=None)
    accept_cluster.set_defaults(command="data accept-cluster")
    converge = data_sub.add_parser(
        "converge",
        help="Run the full-cell 64/96/128 real-MPB resolution gate.",
        description="Re-solve a stratified subset of both full-cell lanes at higher resolutions.",
    )
    converge.add_argument("--config", type=Path, required=True)
    converge.add_argument("--run-dir", type=Path, default=None)
    converge.add_argument(
        "--skip-environment-check",
        action="store_true",
        help="Test-only escape hatch; results are never acceptance eligible.",
    )
    converge.set_defaults(command="data converge")
    finalize = data_sub.add_parser(
        "finalize",
        help="Wait for two fixed full-cell pilot runs, then converge and promote.",
        description="Fail closed unless both real-MPB pilots and the resolution gate pass.",
    )
    finalize.add_argument("--graded-run-dir", type=Path, required=True)
    finalize.add_argument("--binary-run-dir", type=Path, required=True)
    finalize.add_argument("--convergence-config", type=Path, required=True)
    finalize.add_argument("--combined-manifest", type=Path, required=True)
    finalize.add_argument("--poll-seconds", type=float, default=60.0)
    finalize.add_argument("--timeout-hours", type=float, default=8.0)
    finalize.set_defaults(command="data finalize")
    inspect = data_sub.add_parser(
        "inspect",
        help="Inspect a dataset schema without running solvers.",
        description="Inspect a dataset schema without running solvers.",
    )
    inspect.add_argument("--path", type=Path, required=True)
    inspect.set_defaults(command="data inspect")
    convert = data_sub.add_parser(
        "convert",
        help="Convert a dataset between registered representations.",
        description="Convert a dataset between registered representations.",
    )
    convert.add_argument("--config", type=Path, required=True)
    convert.set_defaults(command="data convert")
    make_smoke = data_sub.add_parser(
        "make-smoke",
        help="Generate a deterministic non-physical dataset for clean-clone smoke tests.",
        description=(
            "Generate a configuration-shaped synthetic dataset. The output is explicitly "
            "not MPB/Meep data and cannot support scientific claims."
        ),
    )
    make_smoke.add_argument("--config", type=Path, required=True)
    make_smoke.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing synthetic dataset and manifest explicitly.",
    )
    make_smoke.set_defaults(command="data make-smoke")


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command == "data inspect":
        print(json.dumps(infer_dataset_schema(ns.path), ensure_ascii=False, indent=2))
        return True
    if ns.command == "data make-smoke":
        from phc2d.data.synthetic_smoke import make_synthetic_smoke_dataset

        payload = make_synthetic_smoke_dataset(ns.config, overwrite=ns.overwrite)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return True
    if ns.command == "data generate":
        from phc2d.data.full_cell_generation import run_full_cell_generation

        worker_override: int | str | None = ns.workers
        if isinstance(worker_override, str) and worker_override.lower() != "auto":
            try:
                worker_override = int(worker_override)
            except ValueError as exc:
                raise SystemExit("--workers must be a positive integer or 'auto'.") from exc
        if isinstance(worker_override, int) and worker_override <= 0:
            raise SystemExit("--workers must be positive.")
        plan_worker_override: int | str | None = ns.plan_workers
        if isinstance(plan_worker_override, str) and plan_worker_override.lower() != "auto":
            try:
                plan_worker_override = int(plan_worker_override)
            except ValueError as exc:
                raise SystemExit("--plan-workers must be a positive integer or 'auto'.") from exc
        if isinstance(plan_worker_override, int) and plan_worker_override <= 0:
            raise SystemExit("--plan-workers must be positive.")
        print(
            run_full_cell_generation(
                ns.config,
                run_dir=ns.run_dir,
                max_samples=ns.max_samples,
                workers=worker_override,
                plan_workers=plan_worker_override,
                skip_environment_check=ns.skip_environment_check,
            )
        )
        return True
    if ns.command == "data prepare-full-cell-plan":
        from phc2d.data.full_cell_generation import prepare_full_cell_sample_plan

        plan_worker_override: int | str | None = ns.workers
        if isinstance(plan_worker_override, str) and plan_worker_override.lower() != "auto":
            try:
                plan_worker_override = int(plan_worker_override)
            except ValueError as exc:
                raise SystemExit("--workers must be a positive integer or 'auto'.") from exc
        if isinstance(plan_worker_override, int) and plan_worker_override <= 0:
            raise SystemExit("--workers must be positive.")
        print(
            prepare_full_cell_sample_plan(
                ns.config,
                run_dir=ns.run_dir,
                workers=plan_worker_override,
            )
        )
        return True
    if ns.command == "data accept-cluster":
        from phc2d.data.full_cell_acceptance import run_cluster_wsl_acceptance

        print(
            run_cluster_wsl_acceptance(
                ns.dataset,
                ns.config,
                run_dir=ns.run_dir,
            )
        )
        return True
    if ns.command == "data converge":
        from phc2d.data.full_cell_convergence import run_full_cell_resolution_convergence

        print(
            run_full_cell_resolution_convergence(
                ns.config,
                run_dir=ns.run_dir,
                skip_environment_check=ns.skip_environment_check,
            )
        )
        return True
    if ns.command == "data finalize":
        from phc2d.data.full_cell_convergence import await_full_cell_pilots_and_finalize

        print(
            await_full_cell_pilots_and_finalize(
                graded_run_dir=ns.graded_run_dir,
                binary_run_dir=ns.binary_run_dir,
                convergence_config=ns.convergence_config,
                combined_manifest=ns.combined_manifest,
                poll_interval_s=ns.poll_seconds,
                timeout_s=ns.timeout_hours * 60.0 * 60.0,
            )
        )
        return True
    return False
