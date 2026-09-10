"""Plot command parser and runtime adapter.

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

from phc2d.data import load_forward_dataset
from phc2d.plot import plot_band_overlay, plot_field_map, plot_medium_map, plot_metric_report


def run_plot(argv: Sequence[str]) -> None:
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
    surrogate_physics = sub.add_parser(
        "full-cell-surrogate-physics",
        help="Compare frozen Full-Wh surrogate spectra and real-solver response sensitivity.",
    )
    surrogate_physics.add_argument("--config", type=Path, required=True)
    surrogate_physics.add_argument("--run-dir", type=Path, default=None)
    surrogate_physics.add_argument(
        "--stage",
        choices=["spectral", "physics", "render", "all"],
        default="spectral",
    )
    surrogate_physics.add_argument(
        "--physics-quantiles",
        type=float,
        nargs="+",
        default=None,
        help="Optional quantiles to solve, for example 0.5 for the p50 preflight.",
    )

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
    elif ns.kind == "full-cell-surrogate-physics":
        from phc2d.plot.full_cell_surrogate_physics import build_full_cell_surrogate_physics

        result = build_full_cell_surrogate_physics(
            ns.config,
            run_dir=ns.run_dir,
            stage=ns.stage,
            physics_quantiles=ns.physics_quantiles,
        )
        print(result)
        return
    else:
        raise ValueError(f"Unknown plot kind: {ns.kind}")
    print(result.path)


def configure(sub: Any) -> None:
    """Add plot commands while preserving their remainder-style parser."""

    plot = sub.add_parser("plot", help="Plot medium, band, metric, or field artifacts.")
    plot.add_argument("args", nargs=argparse.REMAINDER)
    plot.set_defaults(command="plot")


def run(ns: Any) -> bool:
    """Run a parsed plot command."""

    if ns.command != "plot":
        return False
    run_plot(ns.args)
    return True
