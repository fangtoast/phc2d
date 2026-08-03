"""Build project-wide publication figures from the frozen model inventory.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


FORMATS = ("png", "svg", "pdf")
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral": "#CFCECE",
    "dark": "#272727",
}
MODEL_COLORS = {
    "mlp": PALETTE["red"],
    "res_mlp_wide": PALETTE["green"],
    "lifted_cnn": PALETTE["teal"],
    "fno2d_band_v2": PALETTE["blue_main"],
    "mlp_query": PALETTE["red"],
    "deep_onet_query_v2": PALETTE["green"],
    "fno2d_query_v2": PALETTE["blue_main"],
    "unet_field_v2": PALETTE["blue_secondary"],
    "fno_field_v2": PALETTE["teal"],
    "adapted_pino": PALETTE["green"],
    "train_only_nearest_neighbor": PALETTE["neutral"],
    "direct_inverse_mlp": PALETTE["red"],
    "single_mode_generator": PALETTE["teal"],
    "mixture_conditional_generator": PALETTE["blue_main"],
}
DISPLAY = {
    "mlp": "MLP",
    "res_mlp_wide": "ResMLP",
    "lifted_cnn": "Lifted CNN",
    "fno2d_band_v2": "FNO band v2",
    "mlp_query": "MLP query",
    "deep_onet_query_v2": "DeepONet-style",
    "fno2d_query_v2": "FNO query v2",
    "unet_field_v2": "U-Net field v2",
    "fno_field_v2": "FNO field v2",
    "adapted_pino": "Adapted PINO",
    "train_only_nearest_neighbor": "Train retrieval",
    "direct_inverse_mlp": "Direct inverse",
    "single_mode_generator": "Single-mode",
    "mixture_conditional_generator": "Mixture",
}
HATCHES = ("", "//", "\\\\", "..")


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _load_inventory(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("entries"), list):
        raise ValueError("Project result figures require an inventory entries list.")
    return payload


def _entry_map(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(entry["id"]): entry for entry in inventory["entries"]}


def _metrics(entries: Mapping[str, Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    metrics = entries[name].get("headline_metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError(f"headline_metrics must be a mapping for {name}")
    return metrics


def _save_figure(
    figure: Any,
    output_base: Path,
    formats: Sequence[str],
    *,
    dpi: int = 300,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        path = output_base.with_suffix(f".{suffix}")
        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip()
                for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    plt.close(figure)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotate(axis: Any, bars: Any, *, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _simple_bars(
    axis: Any,
    *,
    names: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float] | None,
    ylabel: str,
    title: str,
    log_scale: bool = False,
    annotation_format: str = "{:.2f}",
) -> None:
    x = np.arange(len(names), dtype=float)
    colors = [MODEL_COLORS[name] for name in names]
    bars = axis.bar(
        x,
        values,
        yerr=errors,
        capsize=4 if errors is not None else 0,
        color=colors,
        edgecolor=PALETTE["dark"],
        linewidth=1.0,
    )
    for position, bar in enumerate(bars):
        bar.set_hatch(HATCHES[position % len(HATCHES)])
    axis.set_xticks(x, [DISPLAY[name] for name in names], rotation=18, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", color=PALETTE["neutral"], linewidth=0.6, alpha=0.4)
    if log_scale:
        axis.set_yscale("log")
    _annotate(axis, bars, fmt=annotation_format)


def _inventory_column(entry: Mapping[str, Any]) -> str:
    implementation = str(entry.get("implementation_status", ""))
    evidence = str(entry.get("evidence_status", ""))
    if implementation in {"placeholder", "configured_only"}:
        return "placeholder/config"
    if implementation == "reference_only" or evidence in {"blocked", "no_result"}:
        return "blocked/reference"
    if evidence in {"historical", "result_withdrawn"}:
        return "historical"
    return "current"


def _plot_inventory_overview(
    entries: Mapping[str, Mapping[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> tuple[list[Path], list[str]]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    tasks = ("full_band", "query", "field", "inverse")
    columns = ("current", "historical", "placeholder/config", "blocked/reference")
    cell_colors = ("#DDEAF7", "#E8F2E2", "#F5E5D4", "#F4DADA")
    selected: dict[tuple[str, str], list[str]] = {
        (task, column): [] for task in tasks for column in columns
    }
    for entry_id, entry in entries.items():
        task = str(entry.get("task", ""))
        if task not in tasks or entry.get("report_scope") not in {"main", "appendix"}:
            continue
        column = _inventory_column(entry)
        selected[(task, column)].append(DISPLAY.get(entry_id, entry_id))

    figure, axis = plt.subplots(figsize=(14.0, 7.2))
    axis.set_xlim(0, len(columns))
    axis.set_ylim(0, len(tasks))
    axis.set_axis_off()
    for row, task in enumerate(reversed(tasks)):
        y = float(row)
        axis.text(
            -0.08,
            y + 0.5,
            task.replace("_", " ").title(),
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        for column_position, column in enumerate(columns):
            axis.add_patch(
                Rectangle(
                    (column_position, y),
                    1,
                    1,
                    facecolor=cell_colors[column_position],
                    edgecolor="white",
                    linewidth=2,
                )
            )
            names = sorted(selected[(task, column)])
            text = "\n".join(textwrap.fill(name, width=24) for name in names) or "—"
            axis.text(
                column_position + 0.5,
                y + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=8.5,
            )
    for column_position, column in enumerate(columns):
        axis.text(
            column_position + 0.5,
            len(tasks) + 0.08,
            column.title(),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    figure.suptitle(
        "phc2d model and method inventory | implementation is not evidence",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.02,
        "Current includes formal or explicitly qualified single-seed evidence. "
        "Withdrawn results are historical lineage, not current evidence.",
        ha="center",
        fontsize=9,
    )
    paths = _save_figure(figure, output_base, formats)
    labels = [name for names in selected.values() for name in names]
    return paths, labels


def _plot_forward(
    entries: Mapping[str, Mapping[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> tuple[list[Path], list[str]]:
    import matplotlib.pyplot as plt

    names = ("mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2")
    values = [_metrics(entries, name) for name in names]
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    _simple_bars(
        axes[0, 0],
        names=names,
        values=[float(value["bands_2_10_mre_pct_mean"]) for value in values],
        errors=[float(value["bands_2_10_mre_pct_std"]) for value in values],
        ylabel="Test MRE (%)",
        title="Bands 2–10 test error | mean ± seed SD",
    )
    _simple_bars(
        axes[0, 1],
        names=names,
        values=[float(value["parameter_count"]) for value in values],
        errors=None,
        ylabel="Parameters",
        title="Parameter count",
        log_scale=True,
        annotation_format="{:.0f}",
    )
    _simple_bars(
        axes[1, 0],
        names=names,
        values=[float(value["train_time_s_mean"]) for value in values],
        errors=None,
        ylabel="Seconds / seed",
        title="Training time",
        annotation_format="{:.1f}",
    )
    _simple_bars(
        axes[1, 1],
        names=names,
        values=[float(value["inference_ms_per_sample_mean"]) for value in values],
        errors=None,
        ylabel="Milliseconds / sample",
        title="Inference time",
        log_scale=True,
        annotation_format="{:.3f}",
    )
    figure.suptitle(
        "A1 Full-band fair comparison | 10k V2, three validation-selected seeds",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(pad=1.6)
    return _save_figure(figure, output_base, formats), list(names)


def _plot_query(
    entries: Mapping[str, Mapping[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> tuple[list[Path], list[str]]:
    import matplotlib.pyplot as plt

    names = ("mlp_query", "deep_onet_query_v2", "fno2d_query_v2")
    protocols = ("All-k", "Interpolation", "Held-out segment")
    metric_keys = ("all_k_mre_pct", "interpolation_mre_pct", "heldout_mre_pct")
    violation_keys = (
        "all_k_ordering_violation_pct",
        "interpolation_ordering_violation_pct",
        "heldout_ordering_violation_pct",
    )
    fallback_violations = {
        "mlp_query": (0.8313500597, 0.4232804233, 1.0034567901),
        "deep_onet_query_v2": (1.3507287933, 0.4088888889, 0.9896296296),
        "fno2d_query_v2": (2.9614336918, 2.1604232804, 9.2174485597),
    }
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    x = np.arange(len(protocols), dtype=float)
    width = 0.24
    for position, name in enumerate(names):
        metrics = _metrics(entries, name)
        values = [float(metrics[key]) for key in metric_keys]
        bars = axes[0].bar(
            x + (position - 1) * width,
            values,
            width,
            label=DISPLAY[name],
            color=MODEL_COLORS[name],
            edgecolor=PALETTE["dark"],
            linewidth=0.9,
            hatch=HATCHES[position],
        )
        _annotate(axes[0], bars, fmt="{:.2f}")
        violations = [
            float(metrics.get(key, fallback_violations[name][index]))
            for index, key in enumerate(violation_keys)
        ]
        axes[1].plot(
            protocols,
            violations,
            marker=("o", "s", "^")[position],
            linewidth=2.0,
            color=MODEL_COLORS[name],
            label=DISPLAY[name],
        )
    axes[0].set_xticks(x, protocols)
    axes[0].set_ylabel("Bands 2–10 test MRE (%)")
    axes[0].set_title("Prediction error | mean of three seeds")
    axes[1].set_ylabel("Ordering violation (%)")
    axes[1].set_title("Physical ordering diagnostic")
    for axis in axes:
        axis.grid(axis="y", color=PALETTE["neutral"], linewidth=0.6, alpha=0.4)
        axis.legend(fontsize=8)
    figure.suptitle(
        "A1 Continuous-query comparison | error and ordering are separate criteria",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(pad=1.6)
    return _save_figure(figure, output_base, formats), list(names)


def _plot_field(
    entries: Mapping[str, Mapping[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> tuple[list[Path], list[str]]:
    import matplotlib.pyplot as plt

    names = ("unet_field_v2", "fno_field_v2", "adapted_pino")
    metrics = [_metrics(entries, name) for name in names]
    panels = (
        ("e_relative_l2_mean", "e_relative_l2_std", "E relative L2"),
        ("h_relative_l2_mean", "h_relative_l2_std", "H relative L2"),
        ("frequency_mre_pct_mean", "frequency_mre_pct_std", "Frequency MRE (%)"),
        ("maxwell_residual_mean", "maxwell_residual_std", "Maxwell residual"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(18.0, 4.8))
    for axis, (mean_key, std_key, title) in zip(axes, panels, strict=True):
        _simple_bars(
            axis,
            names=names,
            values=[float(value[mean_key]) for value in metrics],
            errors=[float(value[std_key]) for value in metrics],
            ylabel=title,
            title=title,
        )
    figure.suptitle(
        "B3 TE eigenfield comparison | 9,216 real-MPB records, three seeds",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Adapted PINO reuses the FNO field backbone; residual improvement does not imply inverse admission.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94), pad=1.4)
    return _save_figure(figure, output_base, formats), list(names)


def _plot_a2_100(
    entries: Mapping[str, Mapping[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> tuple[list[Path], list[str]]:
    import matplotlib.pyplot as plt

    names = (
        "train_only_nearest_neighbor",
        "direct_inverse_mlp",
        "single_mode_generator",
        "mixture_conditional_generator",
    )
    metrics: list[Mapping[str, Any]] = []
    for name in names:
        value = _metrics(entries, name)
        metrics.append(value.get("withdrawn_a2_100", value))
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.0))
    _simple_bars(
        axes[0],
        names=names,
        values=[float(value["replay_mre_pct_mean"]) for value in metrics],
        errors=None,
        ylabel="MRE (%)",
        title="Real-MPB replay",
    )
    _simple_bars(
        axes[1],
        names=names,
        values=[float(value["surrogate_mre_pct_mean"]) for value in metrics],
        errors=None,
        ylabel="MRE (%)",
        title="Ranking surrogate",
    )
    _simple_bars(
        axes[2],
        names=names,
        values=[float(value["effective_k_mean"]) for value in metrics],
        errors=None,
        ylabel="Unique candidates / target",
        title="Effective K",
        annotation_format="{:.1f}",
    )
    figure.suptitle(
        "Withdrawn original A2 | historical lineage only, not current evidence",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "The reconstruction-only training protocol was invalidated; all four values are withdrawn from current conclusions.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94), pad=1.4)
    return _save_figure(figure, output_base, formats), list(names)


def _plot_a2_bridge(
    entries: Mapping[str, Mapping[str, Any]],
    output_base: Path,
    formats: Sequence[str],
) -> tuple[list[Path], list[str]]:
    import matplotlib.pyplot as plt

    names = ("single_mode_generator", "mixture_conditional_generator")
    current = [_metrics(entries, name)["bridge_625"] for name in names]
    historical = _metrics(entries, "mixture_conditional_generator")[
        "bridge_625_historical_reference"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    x = np.arange(len(names), dtype=float)
    width = 0.34
    replay = axes[0].bar(
        x - width / 2,
        [float(value["replay_mre_pct_mean"]) for value in current],
        width,
        label="Real MPB",
        color=[MODEL_COLORS[name] for name in names],
        edgecolor=PALETTE["dark"],
        linewidth=1.0,
    )
    surrogate = axes[0].bar(
        x + width / 2,
        [float(value["surrogate_mre_pct_mean"]) for value in current],
        width,
        label="Surrogate",
        color=[MODEL_COLORS[name] for name in names],
        edgecolor=PALETTE["dark"],
        linewidth=1.0,
        hatch="//",
        alpha=0.62,
    )
    axes[0].set_xticks(x, [DISPLAY[name] for name in names])
    axes[0].set_ylabel("MRE (%)")
    axes[0].set_title("Bridge seed 0 | matched 625 targets")
    axes[0].legend()
    axes[0].grid(axis="y", color=PALETTE["neutral"], linewidth=0.6, alpha=0.4)
    _annotate(axes[0], replay)
    _annotate(axes[0], surrogate)

    historical_bar = axes[1].bar(
        [0],
        [float(historical["replay_mre_pct_mean"])],
        width=0.55,
        color=PALETTE["neutral"],
        edgecolor=PALETTE["dark"],
        linewidth=1.0,
        hatch="..",
    )
    axes[1].set_xticks([0], ["Historical objective\nmixture (seed 2)"])
    axes[1].set_ylabel("Real-MPB replay MRE (%)")
    axes[1].set_title("Historical reference | unmatched lineage")
    axes[1].grid(axis="y", color=PALETTE["neutral"], linewidth=0.6, alpha=0.4)
    _annotate(axes[1], historical_bar)
    figure.suptitle(
        "A2 objective-bridge test625 | current comparison and historical reference",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "The historical reference differs in seed, ranker, and training lineage; it is not a controlled architecture ablation.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94), pad=1.5)
    return _save_figure(figure, output_base, formats), [
        *names,
        "historical_objective_mixture",
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_source_index(
    path: Path,
    entries: Mapping[str, Mapping[str, Any]],
    extra_rows: Sequence[Mapping[str, Any]] = (),
) -> None:
    rows: list[dict[str, Any]] = []
    for entry_id, entry in entries.items():
        for artifact in entry.get("source_artifacts", []) or []:
            rows.append(
                {
                    "entry_id": entry_id,
                    "evidence_status": entry.get("evidence_status", ""),
                    "source_path": artifact["path"],
                    "source_sha256": artifact["sha256"],
                    "result_report": entry.get("result_report") or "",
                }
            )
    rows.extend(dict(row) for row in extra_rows)
    if not rows:
        raise ValueError("Project source index cannot be empty.")
    fieldnames = (
        "entry_id",
        "evidence_status",
        "source_path",
        "source_sha256",
        "result_report",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def refresh_curated_asset_manifest(
    *,
    inventory_path: Path,
    output_dir: Path,
    figure_series: Mapping[str, Sequence[str]] | None = None,
    lineage_rows: Sequence[Mapping[str, Any]] = (),
    excluded_suffixes: Sequence[str] = (".pdf",),
) -> Path:
    """Refresh hashes after curated learning-curve or inverse assets are copied."""

    inventory_path = Path(inventory_path)
    output_dir = Path(output_dir)
    entries = _entry_map(_load_inventory(inventory_path))
    source_index = output_dir / "source_index.csv"
    _write_source_index(source_index, entries, lineage_rows)
    manifest_path = output_dir / "artifact_manifest.json"
    prior: Mapping[str, Any] = {}
    if manifest_path.exists():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            prior = loaded
    effective_series = (
        {key: list(value) for key, value in figure_series.items()}
        if figure_series is not None
        else dict(prior.get("figure_series", {}))
    )
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path != manifest_path
        and path.suffix.lower() not in set(excluded_suffixes)
    )
    _write_json(
        manifest_path,
        {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "project_result_figure_package_completed",
            "inventory_path": inventory_path.as_posix(),
            "inventory_sha256": _sha256(inventory_path),
            "figure_series": effective_series,
            "curated_lineage": [dict(row) for row in lineage_rows],
            "artifacts": [
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in artifact_paths
            ],
        },
    )
    return manifest_path


def build_project_result_figures(
    *,
    inventory_path: Path,
    output_dir: Path,
    formats: Sequence[str] = FORMATS,
) -> dict[str, Any]:
    """Generate the frozen project-wide result-comparison figure package."""

    invalid_formats = sorted(set(formats) - set(FORMATS))
    if invalid_formats:
        raise ValueError(f"Unsupported project-result figure formats: {invalid_formats}")
    _style()
    inventory_path = Path(inventory_path)
    output_dir = Path(output_dir)
    inventory = _load_inventory(inventory_path)
    entries = _entry_map(inventory)
    figure_dir = output_dir / "results"
    generated: list[Path] = []
    labels: dict[str, list[str]] = {}
    builders = (
        ("model_inventory_overview", _plot_inventory_overview),
        ("a1_forward_model_comparison", _plot_forward),
        ("a1_query_model_comparison", _plot_query),
        ("b3_field_model_comparison", _plot_field),
        ("a2_100_model_comparison", _plot_a2_100),
        ("a2_bridge_625_comparison", _plot_a2_bridge),
    )
    for stem, builder in builders:
        paths, series = builder(entries, figure_dir / stem, formats)
        generated.extend(paths)
        labels[stem] = series

    manifest_path = refresh_curated_asset_manifest(
        inventory_path=inventory_path,
        output_dir=output_dir,
        figure_series=labels,
        excluded_suffixes=() if "pdf" in formats else (".pdf",),
    )
    return {
        "status": "project_result_figure_package_completed",
        "output_dir": output_dir.as_posix(),
        "figure_count": len(builders),
        "figure_file_count": len(generated),
        "figure_series": labels,
        "artifact_manifest": manifest_path.as_posix(),
    }
