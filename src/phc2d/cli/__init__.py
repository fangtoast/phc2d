"""Unified command line for the phc2d workspace.

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

import numpy as np

from phc2d.data import infer_dataset_schema, load_forward_dataset
from phc2d.plot import plot_band_overlay, plot_field_map, plot_medium_map, plot_metric_report


def _not_migrated(command: str) -> None:
    raise SystemExit(
        f"`phc2d {command}` is reserved for the unified pipeline, but the executable "
        "backend has not been migrated into src/phc2d yet."
    )


def _plot(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="phc2d plot",
        description="Plot medium, band, metric, or field artifacts through phc2d.plot.",
    )
    sub = parser.add_subparsers(dest="kind", required=True)

    medium = sub.add_parser("medium", help="Plot one rho sample from a forward dataset.")
    medium.add_argument("--data", type=Path, required=True)
    medium.add_argument("--index", type=int, default=0)
    medium.add_argument("--output", type=Path, required=True)
    medium.add_argument("--mode", choices=["binary", "epsilon"], default="binary")
    medium.add_argument("--title", type=str, default="")

    bands = sub.add_parser("bands", help="Plot target/prediction band overlay from .npy arrays.")
    bands.add_argument("--target", type=Path, required=True)
    bands.add_argument("--prediction", type=Path, default=None)
    bands.add_argument("--output", type=Path, required=True)
    bands.add_argument("--title", type=str, default="Band overlay")

    metrics = sub.add_parser("metrics", help="Plot scalar metrics from JSON.")
    metrics.add_argument("--json", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--title", type=str, default="Metric report")

    field = sub.add_parser("field", help="Plot one scalar field map from a .npy array.")
    field.add_argument("--array", type=Path, required=True)
    field.add_argument("--output", type=Path, required=True)
    field.add_argument("--title", type=str, default="Field map")

    learning = sub.add_parser(
        "learning-curves",
        help="Build A1/A2/P4/P5/P6 learning curves from frozen histories.",
    )
    learning.add_argument("--a1-forward-run", type=Path, required=True)
    learning.add_argument(
        "--a1-mlp-run",
        type=Path,
        default=None,
        help="Optional separate A1.1 mlp run to include in the A1 panels.",
    )
    learning.add_argument("--a1-query-run", type=Path, required=True)
    learning.add_argument("--a2-original-run", type=Path, required=True)
    learning.add_argument("--a2-bridge-run", type=Path, required=True)
    learning.add_argument("--p45-run", type=Path, required=True)
    learning.add_argument("--p6-run", type=Path, required=True)
    learning.add_argument(
        "--field-b3-run",
        type=Path,
        default=None,
        help="Optional B3 U-Net/FNO run used to add supervised field learning curves.",
    )
    learning.add_argument("--output-dir", type=Path, required=True)
    learning.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=["png", "svg"],
    )
    project_results = sub.add_parser(
        "project-results",
        help="Build project-wide model and result figures from a frozen inventory.",
    )
    project_results.add_argument("--inventory", type=Path, required=True)
    project_results.add_argument("--output-dir", type=Path, required=True)
    project_results.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "svg", "pdf"],
        default=["png", "svg", "pdf"],
    )
    full_cell_gallery = sub.add_parser(
        "full-cell-gallery",
        help="Build traceable full-cell dielectric, band, and real-MPB field plates.",
    )
    full_cell_gallery.add_argument("--config", type=Path, required=True)
    full_cell_gallery.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory; defaults to runs/{date}/{experiment}_{timestamp}.",
    )
    full_cell_gallery.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional smoke-only limit applied after deterministic 16-sample selection.",
    )
    full_cell_scattering = sub.add_parser(
        "full-cell-scattering-gallery",
        help="Run finite-array Meep near/far-field responses for selected full-cell samples.",
    )
    full_cell_scattering.add_argument("--config", type=Path, required=True)
    full_cell_scattering.add_argument("--run-dir", type=Path, default=None)
    full_cell_scattering.add_argument("--max-samples-per-dataset", type=int, default=None)

    ns = parser.parse_args(list(argv))
    if ns.kind == "medium":
        bundle = load_forward_dataset(ns.data)
        result = plot_medium_map(bundle.rho[int(ns.index)], ns.output, mode=ns.mode, title=ns.title)
    elif ns.kind == "bands":
        pred = np.load(ns.prediction) if ns.prediction is not None else None
        result = plot_band_overlay(np.load(ns.target), ns.output, prediction=pred, title=ns.title)
    elif ns.kind == "metrics":
        payload: dict[str, Any] = json.loads(ns.json.read_text(encoding="utf-8"))
        result = plot_metric_report(payload, ns.output, title=ns.title)
    elif ns.kind == "field":
        result = plot_field_map(np.load(ns.array), ns.output, title=ns.title)
    elif ns.kind == "learning-curves":
        from phc2d.plot.training_curves import build_learning_curve_package

        result = build_learning_curve_package(
            a1_forward_run=ns.a1_forward_run,
            a1_mlp_run=ns.a1_mlp_run,
            a1_query_run=ns.a1_query_run,
            a2_original_run=ns.a2_original_run,
            a2_bridge_run=ns.a2_bridge_run,
            p45_run=ns.p45_run,
            p6_run=ns.p6_run,
            field_b3_run=ns.field_b3_run,
            output_dir=ns.output_dir,
            formats=ns.formats,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    elif ns.kind == "project-results":
        from phc2d.plot.project_results import build_project_result_figures

        result = build_project_result_figures(
            inventory_path=ns.inventory,
            output_dir=ns.output_dir,
            formats=ns.formats,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    elif ns.kind == "full-cell-gallery":
        from phc2d.plot.full_cell_gallery import build_full_cell_sample_gallery

        result = build_full_cell_sample_gallery(
            ns.config,
            run_dir=ns.run_dir,
            max_samples_per_dataset=ns.max_samples_per_dataset,
        )
    elif ns.kind == "full-cell-scattering-gallery":
        from phc2d.plot.full_cell_scattering import build_full_cell_scattering_gallery

        result = build_full_cell_scattering_gallery(
            ns.config,
            run_dir=ns.run_dir,
            max_samples_per_dataset=ns.max_samples_per_dataset,
        )
        print(result)
        return
    else:
        raise ValueError(f"Unknown plot kind: {ns.kind}")
    print(result.path)


def build_parser() -> argparse.ArgumentParser:
    """Build the phc2d top-level parser without importing heavy training backends."""
    parser = argparse.ArgumentParser(
        prog="phc2d",
        description="Unified 2D photonic-crystal workflow CLI.",
    )
    parser.add_argument("--version", action="version", version="phc2d 0.2.0")
    sub = parser.add_subparsers(dest="domain", required=True)

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
        "--skip-environment-check",
        action="store_true",
        help="Test-only escape hatch; never use this to claim accepted MPB validation.",
    )
    generate.set_defaults(command="data generate")
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

    forward = sub.add_parser("forward", help="Forward surrogate training and evaluation.")
    forward_sub = forward.add_subparsers(dest="action", required=True)
    forward_train = forward_sub.add_parser(
        "train",
        help="Train a registered forward surrogate.",
        description="Train a registered forward surrogate.",
    )
    forward_train.add_argument("--config", type=Path, required=True)
    forward_train.set_defaults(command="forward train")
    forward_eval = forward_sub.add_parser(
        "eval",
        help="Evaluate a frozen forward surrogate checkpoint.",
        description="Evaluate a frozen forward surrogate checkpoint.",
    )
    forward_eval.add_argument("--config", type=Path, required=True)
    forward_eval.set_defaults(command="forward eval")
    forward_benchmark = forward_sub.add_parser(
        "benchmark",
        help="Run a small representation-model benchmark matrix.",
        description="Run a small representation-model benchmark matrix and write leaderboard + per-band summaries.",
    )
    forward_benchmark.add_argument("--config", type=Path, required=True)
    forward_benchmark.set_defaults(command="forward benchmark")

    query = sub.add_parser("query", help="Query-form band surrogate audit and benchmarks.")
    query_sub = query.add_subparsers(dest="action", required=True)
    query_audit = query_sub.add_parser(
        "audit",
        help="Audit query dataset schema, split lineage, and k-path metadata.",
        description="Audit query dataset schema, split lineage, and k-path metadata.",
    )
    query_audit.add_argument("--config", type=Path, required=True)
    query_audit.set_defaults(command="query audit")
    query_preflight = query_sub.add_parser(
        "preflight",
        help="Run GPU query model matrix preflight.",
        description="Run GPU query model matrix preflight.",
    )
    query_preflight.add_argument("--config", type=Path, required=True)
    query_preflight.set_defaults(command="query preflight")
    query_benchmark = query_sub.add_parser(
        "benchmark",
        help="Run formal query-form benchmark when gates allow it.",
        description="Run formal query-form benchmark when gates allow it.",
    )
    query_benchmark.add_argument("--config", type=Path, required=True)
    query_benchmark.set_defaults(command="query benchmark")
    query_data_scaling = query_sub.add_parser(
        "data-scaling",
        help="Run guarded query-form data scaling over configured V2 datasets.",
        description="Run guarded query-form data scaling over configured V2 datasets.",
    )
    query_data_scaling.add_argument("--config", type=Path, required=True)
    query_data_scaling.set_defaults(command="query data-scaling")

    models = sub.add_parser("models", help="Registered forward surrogate model families.")
    models_sub = models.add_subparsers(dest="action", required=True)
    models_list = models_sub.add_parser(
        "list",
        help="List registered ModelSpec entries (read-only registry view).",
        description="List registered ModelSpec entries without importing torch.",
    )
    models_list.set_defaults(command="models list")

    inverse = sub.add_parser("inverse", help="Inverse design workflows.")
    inverse_sub = inverse.add_subparsers(dest="action", required=True)
    inverse_train = inverse_sub.add_parser(
        "train",
        help="Train dispersion inverse design using a frozen surrogate.",
        description="Train dispersion inverse design using a frozen surrogate.",
    )
    inverse_train.add_argument("--config", type=Path, default=None)
    inverse_train.set_defaults(command="inverse train")
    inverse_objective_train = inverse_sub.add_parser(
        "objective-train",
        help="Fine-tune an inverse generator against a differentiable surrogate-side objective.",
        description=(
            "Fine-tune an inverse generator with a differentiable surrogate objective. "
            "This is not MPB/Meep validation."
        ),
    )
    inverse_objective_train.add_argument("--config", type=Path, required=True)
    inverse_objective_train.set_defaults(command="inverse objective-train")
    inverse_sample = inverse_sub.add_parser(
        "sample",
        help="Sample inverse-design candidates from a registered generator.",
        description="Sample inverse-design candidates from a registered generator.",
    )
    inverse_sample.add_argument("--config", type=Path, required=True)
    inverse_sample.set_defaults(command="inverse sample")
    inverse_rerank = inverse_sub.add_parser(
        "rerank",
        help="Rerank existing inverse-design candidates with one or more frozen surrogates.",
        description="Rerank existing inverse candidates with query, forward, or hybrid frozen surrogate strategies.",
    )
    inverse_rerank.add_argument("--config", type=Path, required=True)
    inverse_rerank.set_defaults(command="inverse rerank")
    inverse_eval = inverse_sub.add_parser(
        "eval",
        help="Evaluate inverse candidates with surrogate-side metrics.",
        description="Evaluate inverse candidates with surrogate-side metrics.",
    )
    inverse_eval.add_argument("--config", type=Path, required=True)
    inverse_eval.set_defaults(command="inverse eval")
    inverse_diagnose = inverse_sub.add_parser(
        "diagnose",
        help="Build a gate ledger from existing inverse/replay artifacts.",
        description=(
            "Read existing inverse sample and replay artifacts, then write a diagnostic "
            "ledger without training models or running MPB/Meep."
        ),
    )
    inverse_diagnose.add_argument("--config", type=Path, required=True)
    inverse_diagnose.set_defaults(command="inverse diagnose")
    inverse_design_diagnose = inverse_sub.add_parser(
        "design-diagnose",
        help="Write a target-candidate trace report from existing inverse-design artifacts.",
        description=(
            "Read existing inverse objective-train, sample, and replay artifacts, then "
            "write a trace report without changing models or running MPB/Meep."
        ),
    )
    inverse_design_diagnose.add_argument("--config", type=Path, required=True)
    inverse_design_diagnose.set_defaults(command="inverse design-diagnose")
    inverse_benchmark = inverse_sub.add_parser(
        "benchmark",
        help="Run the resumable A2 single-seed fair inverse benchmark.",
        description=(
            "Run one evidence-gated A2 stage: preflight, train, sample, pilot-replay, "
            "pilot-aggregate, replay, aggregate, surrogate screen/promotion/derivation, "
            "fake-smoke, or real-smoke."
        ),
    )
    inverse_benchmark.add_argument("--config", type=Path, required=True)
    inverse_benchmark.add_argument(
        "--stage",
        choices=(
            "preflight",
            "train",
            "sample",
            "screen",
            "pilot-replay",
            "pilot-aggregate",
            "promote",
            "derive-objective",
            "derive-formal",
            "replay",
            "aggregate",
            "fake-smoke",
            "real-smoke",
        ),
        required=True,
    )
    inverse_benchmark.add_argument("--run-dir", type=Path, default=None)
    inverse_benchmark.set_defaults(command="inverse benchmark")

    replay = sub.add_parser("replay", help="Physics replay and formal validation helpers.")
    replay_sub = replay.add_subparsers(dest="action", required=True)
    replay_mpb = replay_sub.add_parser(
        "mpb",
        help="Run MPB replay through WSL Ubuntu conda meep_env.",
        description="Run MPB replay through WSL Ubuntu conda meep_env.",
    )
    replay_mpb.add_argument("--config", type=Path, default=None)
    replay_mpb.set_defaults(command="replay mpb")

    plot = sub.add_parser("plot", help="Plot medium, band, metric, or field artifacts.")
    plot.add_argument("args", nargs=argparse.REMAINDER)
    plot.set_defaults(command="plot")

    report = sub.add_parser("report", help="Result-report helpers.")
    report_sub = report.add_subparsers(dest="action")
    report_build = report_sub.add_parser("build", help="Build a curated report from one run directory.")
    report_build.add_argument("--run", type=Path, required=True)
    report_build.set_defaults(command="report build")
    report_model_audit = report_sub.add_parser(
        "model-audit",
        help="Audit model inventory, master-report coverage, links, and source hashes.",
        description=(
            "Read a model inventory and master report without importing torch, training models, "
            "or running MPB/Meep."
        ),
    )
    report_model_audit.add_argument("--inventory", type=Path, required=True)
    report_model_audit.add_argument("--report", type=Path, required=True)
    report_model_audit.add_argument(
        "--require-source-artifacts",
        action="store_true",
        help="Fail when ignored runs/source artifacts are unavailable.",
    )
    report_model_audit.set_defaults(command="report model-audit")
    report_docs_audit = report_sub.add_parser(
        "docs-audit",
        help="Audit documentation links, reachability, anchors, and entry metadata.",
        description="Audit Git-tracked project documentation without importing training backends.",
    )
    report_docs_audit.add_argument("--root", type=Path, default=Path("README.md"))
    report_docs_audit.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when documentation contract failures are found.",
    )
    report_docs_audit.set_defaults(command="report docs-audit")
    report.set_defaults(command="report")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the unified CLI."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.command == "plot":
        _plot(ns.args)
        return
    if ns.command == "data inspect":
        print(json.dumps(infer_dataset_schema(ns.path), ensure_ascii=False, indent=2))
        return
    if ns.command == "data make-smoke":
        from phc2d.data.synthetic_smoke import make_synthetic_smoke_dataset

        payload = make_synthetic_smoke_dataset(ns.config, overwrite=ns.overwrite)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
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
        print(
            run_full_cell_generation(
                ns.config,
                run_dir=ns.run_dir,
                max_samples=ns.max_samples,
                workers=worker_override,
                skip_environment_check=ns.skip_environment_check,
            )
        )
        return
    if ns.command == "data accept-cluster":
        from phc2d.data.full_cell_acceptance import run_cluster_wsl_acceptance

        print(
            run_cluster_wsl_acceptance(
                ns.dataset,
                ns.config,
                run_dir=ns.run_dir,
            )
        )
        return
    if ns.command == "data converge":
        from phc2d.data.full_cell_convergence import run_full_cell_resolution_convergence

        print(
            run_full_cell_resolution_convergence(
                ns.config,
                run_dir=ns.run_dir,
                skip_environment_check=ns.skip_environment_check,
            )
        )
        return
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
        return
    if ns.command == "models list":
        from phc2d.models import list_model_specs, model_spec_to_dict

        payload = [model_spec_to_dict(spec) for spec in list_model_specs()]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if ns.command == "forward train":
        from phc2d.forward import run_forward_train

        print(run_forward_train(ns.config))
        return
    if ns.command == "forward eval":
        from phc2d.forward import run_forward_eval

        print(run_forward_eval(ns.config))
        return
    if ns.command == "forward benchmark":
        from phc2d.forward import run_forward_benchmark

        print(run_forward_benchmark(ns.config))
        return
    if ns.command == "query audit":
        from phc2d.query import run_query_audit

        print(run_query_audit(ns.config))
        return
    if ns.command == "query preflight":
        from phc2d.query import run_query_preflight

        print(run_query_preflight(ns.config))
        return
    if ns.command == "query benchmark":
        from phc2d.query import run_query_benchmark

        print(run_query_benchmark(ns.config))
        return
    if ns.command == "query data-scaling":
        from phc2d.query import run_query_data_scaling

        print(run_query_data_scaling(ns.config))
        return
    if ns.command == "replay mpb":
        from phc2d.replay import run_replay_mpb

        if ns.config is None:
            raise SystemExit("`phc2d replay mpb` requires --config <yaml-path>.")
        print(run_replay_mpb(ns.config))
        return
    if ns.command == "inverse train":
        from phc2d.inverse import run_inverse_train

        if ns.config is None:
            raise SystemExit("`phc2d inverse train` requires --config <yaml-path>.")
        print(run_inverse_train(ns.config))
        return
    if ns.command == "inverse objective-train":
        from phc2d.inverse import run_inverse_objective_train

        print(run_inverse_objective_train(ns.config))
        return
    if ns.command == "inverse sample":
        from phc2d.inverse import run_inverse_sample

        print(run_inverse_sample(ns.config))
        return
    if ns.command == "inverse rerank":
        from phc2d.inverse import run_inverse_rerank

        print(run_inverse_rerank(ns.config))
        return
    if ns.command == "inverse eval":
        from phc2d.inverse import run_inverse_eval

        print(run_inverse_eval(ns.config))
        return
    if ns.command == "inverse diagnose":
        from phc2d.inverse import run_inverse_diagnose

        print(run_inverse_diagnose(ns.config))
        return
    if ns.command == "inverse design-diagnose":
        from phc2d.inverse import run_inverse_design_diagnostic

        print(run_inverse_design_diagnostic(ns.config))
        return
    if ns.command == "inverse benchmark":
        from phc2d.inverse import run_a2_benchmark

        print(run_a2_benchmark(ns.config, stage=ns.stage, run_dir=ns.run_dir))
        return
    if ns.command in {"report", "report build"}:
        print("Curated reports live under docs/results; raw outputs live under runs/.")
        return
    if ns.command == "report model-audit":
        from phc2d.reporting import audit_model_report

        payload = audit_model_report(
            inventory_path=ns.inventory,
            report_path=ns.report,
            require_source_artifacts=ns.require_source_artifacts,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if payload["status"] != "pass":
            raise SystemExit(1)
        return
    if ns.command == "report docs-audit":
        from phc2d.reporting.docs_audit import audit_documentation

        payload = audit_documentation(repo_root=Path.cwd(), root_path=ns.root)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if ns.strict and payload["status"] != "pass":
            raise SystemExit(1)
        return
    _not_migrated(ns.command)

__all__ = ["build_parser", "main"]
