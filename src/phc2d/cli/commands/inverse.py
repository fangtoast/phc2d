"""Inverse command parser and lazy runtime dispatch.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def configure(sub: Any) -> None:
    """Add this command domain to the top-level parser."""

    inverse = sub.add_parser("inverse", help="Inverse design workflows.")
    inverse_sub = inverse.add_subparsers(dest="action", required=True)
    inverse_train = inverse_sub.add_parser(
        "train",
        help="Train dispersion inverse design using a frozen surrogate.",
        description="Train dispersion inverse design using a frozen surrogate.",
    )
    inverse_train.add_argument("--config", type=Path, default=None)
    inverse_train.set_defaults(command="inverse train")
    inverse_full_cell_preflight = inverse_sub.add_parser(
        "full-cell-preflight",
        help="Run the 15-case full-cell analytic Stage-A single-batch gate.",
        description=(
            "Run the 15-case full-cell analytic Stage-A gate for data, gradients, and checkpoint restore. "
            "This does not start formal training or MPB/Meep replay."
        ),
    )
    inverse_full_cell_preflight.add_argument("--config", type=Path, required=True)
    inverse_full_cell_preflight.add_argument("--run-dir", type=Path)
    inverse_full_cell_preflight.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Restrict the preflight to one variant; repeat to select multiple variants.",
    )
    inverse_full_cell_preflight.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Restrict the preflight to one method; repeat to select multiple methods.",
    )
    inverse_full_cell_preflight.add_argument(
        "--seed",
        type=int,
        help="Override the configured seed for an isolated concurrency case.",
    )
    inverse_full_cell_preflight.set_defaults(command="inverse full-cell-preflight")
    inverse_full_cell_train = inverse_sub.add_parser(
        "full-cell-train",
        help="Train one full-cell analytic Stage-A variant/method/seed job.",
        description=(
            "Train and validate one supervised full-cell analytic Stage-A job with atomic "
            "checkpoints and exact resume. This does not use a forward surrogate or run MPB/Meep."
        ),
    )
    inverse_full_cell_train.add_argument("--config", type=Path, required=True)
    inverse_full_cell_train.add_argument("--variant", required=True)
    inverse_full_cell_train.add_argument("--method", required=True)
    inverse_full_cell_train.add_argument("--seed", type=int, required=True)
    inverse_full_cell_train.add_argument("--run-dir", type=Path, required=True)
    inverse_full_cell_train.set_defaults(command="inverse full-cell-train")
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


def run(ns: Any) -> bool:
    """Run a parsed command when it belongs to this domain."""

    if ns.command == "inverse train":
        from phc2d.inverse import run_inverse_train

        if ns.config is None:
            raise SystemExit("`phc2d inverse train` requires --config <yaml-path>.")
        print(run_inverse_train(ns.config))
        return True
    if ns.command == "inverse full-cell-preflight":
        from phc2d.inverse import run_full_cell_stage_a_preflight

        print(
            run_full_cell_stage_a_preflight(
                ns.config,
                run_dir=ns.run_dir,
                variants=ns.variants,
                methods=ns.methods,
                seed=ns.seed,
            )
        )
        return True
    if ns.command == "inverse full-cell-train":
        from phc2d.inverse import run_full_cell_stage_a_train

        print(
            run_full_cell_stage_a_train(
                ns.config,
                variant=ns.variant,
                method=ns.method,
                seed=ns.seed,
                run_dir=ns.run_dir,
            )
        )
        return True
    if ns.command == "inverse objective-train":
        from phc2d.inverse import run_inverse_objective_train

        print(run_inverse_objective_train(ns.config))
        return True
    if ns.command == "inverse sample":
        from phc2d.inverse import run_inverse_sample

        print(run_inverse_sample(ns.config))
        return True
    if ns.command == "inverse rerank":
        from phc2d.inverse import run_inverse_rerank

        print(run_inverse_rerank(ns.config))
        return True
    if ns.command == "inverse eval":
        from phc2d.inverse import run_inverse_eval

        print(run_inverse_eval(ns.config))
        return True
    if ns.command == "inverse diagnose":
        from phc2d.inverse import run_inverse_diagnose

        print(run_inverse_diagnose(ns.config))
        return True
    if ns.command == "inverse design-diagnose":
        from phc2d.inverse import run_inverse_design_diagnostic

        print(run_inverse_design_diagnostic(ns.config))
        return True
    if ns.command == "inverse benchmark":
        from phc2d.inverse import run_a2_benchmark

        print(run_a2_benchmark(ns.config, stage=ns.stage, run_dir=ns.run_dir))
        return True
    return False
