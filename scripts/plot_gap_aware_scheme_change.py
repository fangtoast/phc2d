"""Visualize EXP-FWD-014 changes with EXP-PHY-005/006 physical context.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


COLORS = {
    "truth": "#2166AC",
    "prediction": "#C83E3A",
    "baseline": "#2166AC",
    "new": "#C83E3A",
    "neutral": "#666666",
    "grid": "#D9D9D9",
    "safe": "#E7EEF5",
}

SAMPLES = (
    ("binary", 89),
    ("binary", 96),
    ("graded", 62),
    ("graded", 337),
)

# Curated EXP-FWD-014 validation summary. The remote comparison run is the raw
# authority; these values match its current repository result page.
DECISION_METRICS = {
    "binary": {
        "Max-gap band": (36.7, 37.7),
        "Upper-edge k": (21.5, 42.4),
        "Joint band-k": (13.5, 24.4),
    },
    "graded": {
        "Max-gap band": (34.5, 40.9),
        "Upper-edge k": (8.9, 35.7),
        "Joint band-k": (7.7, 22.8),
    },
}

ERROR_METRICS = {
    "binary": {
        "Bands 2-10\nMRE": (1.2223, 1.3867),
        "Work-frequency\nMRE": (16.0545, 15.3047),
        "Gap-width\nMAE": (0.007775, 0.007243),
        "Gap-center\nMAE": (0.088173, 0.094588),
    },
    "graded": {
        "Bands 2-10\nMRE": (1.0126, 1.1407),
        "Work-frequency\nMRE": (17.7614, 15.1620),
        "Gap-width\nMAE": (0.006699, 0.006359),
        "Gap-center\nMAE": (0.091928, 0.084099),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the measured EXP-FWD-014 scheme changes and physical gap-label audit."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory relative to root unless absolute.",
    )
    return parser.parse_args()


def _output_dir(root: Path, requested: Path | None) -> Path:
    if requested is None:
        return root / "docs" / "reports" / "assets" / "gap_aware_scheme_change_20260831"
    return requested if requested.is_absolute() else root / requested


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_gap_rows(root: Path) -> dict[tuple[str, int], list[dict[str, object]]]:
    rows = _read_csv(
        root / "runs" / "20260829" / "EXP-PHY-006_202608291923" / "gap_refinement_samples.csv"
    )
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {key: [] for key in SAMPLES}
    for row in rows:
        key = (row["lane"], int(row["dataset_index"]))
        if key not in grouped:
            continue
        grouped[key].append(
            {
                "pair": int(row["lower_band_one_based"]),
                "coarse_lower": float(row["coarse_lower_edge_frequency"]),
                "coarse_upper": float(row["coarse_upper_edge_frequency"]),
                "coarse_center": float(row["coarse_center"]),
                "coarse_width": float(row["coarse_signed_width"]),
                "refined_lower": float(row["refined_lower_edge_frequency"]),
                "refined_upper": float(row["refined_upper_edge_frequency"]),
                "refined_center": float(row["refined_center"]),
                "refined_width": float(row["refined_signed_width"]),
                "status_changed": row["status_changed"].lower() == "true",
            }
        )
    for key, values in grouped.items():
        values.sort(key=lambda item: int(item["pair"]))
        if len(values) != 9:
            raise ValueError(f"Expected nine gap pairs for {key}, found {len(values)}.")
    return grouped


def _load_formula_x(root: Path) -> dict[tuple[str, int], float]:
    rows = _read_csv(
        root / "runs" / "20260828" / "EXP-PHY-005_202608281751" / "gap_alignment_rows.csv"
    )
    selected: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["lane"], int(row["dataset_index"]))
        if row["split"] == "validation" and row["candidate"] == "x" and key in SAMPLES:
            selected[key] = float(row["new_frequency"])
    missing = set(SAMPLES) - set(selected)
    if missing:
        raise ValueError(f"Missing EXP-PHY-005 x frequencies for {sorted(missing)}.")
    return selected


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.65, zorder=0)
    axis.tick_params(width=0.7, length=3)


def _plot_decision_accuracy(axis: plt.Axes) -> None:
    metric_names = list(DECISION_METRICS["binary"])
    positions = np.arange(6, dtype=np.float64)
    baseline = np.asarray(
        [DECISION_METRICS[lane][name][0] for lane in ("binary", "graded") for name in metric_names]
    )
    new = np.asarray(
        [DECISION_METRICS[lane][name][1] for lane in ("binary", "graded") for name in metric_names]
    )
    width = 0.34
    axis.bar(
        positions - width / 2,
        baseline,
        width,
        color=COLORS["baseline"],
        label="MSE baseline",
        zorder=2,
    )
    axis.bar(
        positions + width / 2,
        new,
        width,
        color=COLORS["new"],
        label="Gap-aware v1",
        zorder=2,
    )
    for x, old, current in zip(positions, baseline, new, strict=True):
        axis.text(x + width / 2, current + 1.0, f"+{current-old:.1f}", ha="center", fontsize=5.8)
    compact_names = ["Max-gap\nband", "Upper-edge\nk", "Joint\nband-k"]
    labels = [f"B\n{name}" for name in compact_names] + [f"G\n{name}" for name in compact_names]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=5.8)
    axis.axvline(2.5, color=COLORS["grid"], linewidth=0.8)
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_ylim(0, 50)
    axis.set_title("Measured decision-location gains", loc="left")
    axis.legend(loc="upper left", ncol=2)
    _style_axis(axis)
    _panel_label(axis, "a")


def _plot_error_changes(axis: plt.Axes) -> None:
    names = list(ERROR_METRICS["binary"])
    positions = np.arange(8, dtype=np.float64)
    changes = np.asarray(
        [
            100.0 * (ERROR_METRICS[lane][name][1] / ERROR_METRICS[lane][name][0] - 1.0)
            for lane in ("binary", "graded")
            for name in names
        ]
    )
    colors = [COLORS["new"] if value > 0 else COLORS["baseline"] for value in changes]
    axis.axhspan(-5.0, 5.0, color=COLORS["safe"], alpha=0.8, zorder=0)
    axis.axhline(0.0, color=COLORS["neutral"], linewidth=0.8, zorder=1)
    axis.bar(positions, changes, color=colors, width=0.66, zorder=2)
    for x, value in zip(positions, changes, strict=True):
        axis.text(
            x,
            value + (0.8 if value >= 0 else -0.8),
            f"{value:+.2f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=5.8,
        )
    compact_names = ["Bands 2-10\nMRE", "Work-freq.\nMRE", "Gap-width\nMAE", "Gap-center\nMAE"]
    labels = [f"B\n{name}" for name in compact_names] + [f"G\n{name}" for name in compact_names]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=5.5)
    axis.axvline(3.5, color=COLORS["grid"], linewidth=0.8)
    axis.set_ylabel("Relative change from MSE baseline (%)")
    axis.set_ylim(-18, 18)
    axis.set_title("Accuracy trade-off introduced by gap-aware v1", loc="left")
    axis.text(
        0.99,
        0.97,
        "Blue: lower error\nRed: higher error",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
    )
    _style_axis(axis)
    _panel_label(axis, "b")


def _add_interval(
    axis: plt.Axes,
    *,
    x: float,
    edge_a: float,
    edge_b: float,
    signed_width: float,
    color: str,
) -> None:
    bottom = min(edge_a, edge_b)
    height = abs(edge_b - edge_a)
    rectangle = Rectangle(
        (x - 0.12, bottom),
        0.24,
        height,
        facecolor=color,
        edgecolor=color,
        linewidth=0.45,
        alpha=0.18 if signed_width >= 0 else 0.10,
        hatch=None if signed_width >= 0 else "////",
        zorder=1,
    )
    axis.add_patch(rectangle)


def _plot_physical_sample(
    axis: plt.Axes,
    *,
    key: tuple[str, int],
    rows: list[dict[str, object]],
    formula_x: float,
    label: str,
) -> None:
    positions = np.arange(1, 10, dtype=np.float64)
    coarse_center = np.asarray([float(row["coarse_center"]) for row in rows])
    refined_center = np.asarray([float(row["refined_center"]) for row in rows])
    for x, row in zip(positions, rows, strict=True):
        _add_interval(
            axis,
            x=x - 0.10,
            edge_a=float(row["coarse_lower"]),
            edge_b=float(row["coarse_upper"]),
            signed_width=float(row["coarse_width"]),
            color=COLORS["prediction"],
        )
        _add_interval(
            axis,
            x=x + 0.10,
            edge_a=float(row["refined_lower"]),
            edge_b=float(row["refined_upper"]),
            signed_width=float(row["refined_width"]),
            color=COLORS["truth"],
        )
    axis.plot(
        positions - 0.10,
        coarse_center,
        color=COLORS["prediction"],
        marker="o",
        markersize=3.0,
        linewidth=1.0,
        label="9x9 coarse estimate",
        zorder=3,
    )
    axis.plot(
        positions + 0.10,
        refined_center,
        color=COLORS["truth"],
        marker="o",
        markersize=3.0,
        linewidth=1.0,
        label="Locally refined MPB reference",
        zorder=4,
    )
    axis.scatter(
        [0.90],
        [formula_x],
        facecolors="white",
        edgecolors=COLORS["prediction"],
        linewidths=1.1,
        marker="D",
        s=25,
        label=r"Formula $x=1/(2n_{\mathrm{eff}})$",
        zorder=5,
    )
    changed = [int(row["pair"]) for row in rows if bool(row["status_changed"])]
    if changed:
        axis.annotate(
            "complete to pseudo",
            xy=(changed[0] + 0.10, refined_center[changed[0] - 1]),
            xytext=(changed[0] - 2.2, refined_center[changed[0] - 1] + 0.08),
            fontsize=5.8,
            color=COLORS["prediction"],
            arrowprops={"arrowstyle": "->", "color": COLORS["prediction"], "lw": 0.7},
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([f"B{i}-B{i+1}" for i in range(1, 10)], rotation=35, ha="right")
    axis.set_ylabel(r"Gap center / frequency $\omega a/(2\pi c)$")
    axis.set_title(f"{key[0].title()} validation | index {key[1]}", loc="left")
    axis.set_xlim(0.45, 9.55)
    _style_axis(axis)
    _panel_label(axis, label)


def _write_source_data(
    path: Path,
    gap_rows: dict[tuple[str, int], list[dict[str, object]]],
    formula_x: dict[tuple[str, int], float],
) -> None:
    fields = [
        "evidence_layer",
        "lane",
        "dataset_index",
        "metric_or_pair",
        "baseline_or_coarse",
        "new_or_refined",
        "relative_change_pct",
        "formula_x_frequency",
        "coarse_lower_edge",
        "coarse_upper_edge",
        "coarse_signed_width",
        "refined_lower_edge",
        "refined_upper_edge",
        "refined_signed_width",
        "status_changed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for lane in ("binary", "graded"):
            for name, (baseline, new) in DECISION_METRICS[lane].items():
                writer.writerow(
                    {
                        "evidence_layer": "EXP-FWD-014 surrogate validation decision metric",
                        "lane": lane,
                        "metric_or_pair": name,
                        "baseline_or_coarse": baseline,
                        "new_or_refined": new,
                        "relative_change_pct": 100.0 * (new / baseline - 1.0),
                    }
                )
            for name, (baseline, new) in ERROR_METRICS[lane].items():
                writer.writerow(
                    {
                        "evidence_layer": "EXP-FWD-014 surrogate validation error metric",
                        "lane": lane,
                        "metric_or_pair": name.replace("\n", " "),
                        "baseline_or_coarse": baseline,
                        "new_or_refined": new,
                        "relative_change_pct": 100.0 * (new / baseline - 1.0),
                    }
                )
        for key in SAMPLES:
            for row in gap_rows[key]:
                writer.writerow(
                    {
                        "evidence_layer": "EXP-PHY-005/006 physical label audit",
                        "lane": key[0],
                        "dataset_index": key[1],
                        "metric_or_pair": f"B{row['pair']}-B{int(row['pair'])+1}",
                        "baseline_or_coarse": row["coarse_center"],
                        "new_or_refined": row["refined_center"],
                        "formula_x_frequency": formula_x[key] if int(row["pair"]) == 1 else "",
                        "coarse_lower_edge": row["coarse_lower"],
                        "coarse_upper_edge": row["coarse_upper"],
                        "coarse_signed_width": row["coarse_width"],
                        "refined_lower_edge": row["refined_lower"],
                        "refined_upper_edge": row["refined_upper"],
                        "refined_signed_width": row["refined_width"],
                        "status_changed": row["status_changed"],
                    }
                )


def _strip_svg_trailing_whitespace(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    path.write_text(cleaned, encoding="utf-8", newline="\n")


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = _output_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gap_rows = _load_gap_rows(root)
    formula_x = _load_formula_x(root)

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 6.1,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.4,
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(7.20, 7.10), constrained_layout=True)
    _plot_decision_accuracy(axes[0, 0])
    _plot_error_changes(axes[0, 1])
    for position, key in enumerate(SAMPLES):
        axis = axes[1 + position // 2, position % 2]
        _plot_physical_sample(
            axis,
            key=key,
            rows=gap_rows[key],
            formula_x=formula_x[key],
            label="cdef"[position],
        )
    axes[1, 0].legend(
        handles=[
            plt.Line2D([], [], color=COLORS["prediction"], marker="o", lw=1.0, label="9x9 coarse estimate"),
            plt.Line2D([], [], color=COLORS["truth"], marker="o", lw=1.0, label="Refined MPB reference"),
            plt.Line2D(
                [],
                [],
                color=COLORS["prediction"],
                marker="D",
                markerfacecolor="white",
                lw=0,
                label=r"Formula $x=1/(2n_{\mathrm{eff}})$",
            ),
            Patch(facecolor="white", edgecolor=COLORS["neutral"], hatch="////", label="Hatched interval: overlap"),
        ],
        loc="upper left",
        fontsize=5.6,
    )
    figure.suptitle(
        "Gap-aware v1 improves decision location but trades away full-spectrum accuracy",
        fontsize=8.5,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.008,
        "a-b: EXP-FWD-014 surrogate validation (seed0, no test).  c-f: independent EXP-PHY-005/006 physical-label audit.",
        ha="center",
        fontsize=6.1,
        color=COLORS["neutral"],
    )
    base = output_dir / "gap_aware_scheme_change_with_physical_reference"
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    _strip_svg_trailing_whitespace(base.with_suffix(".svg"))
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    _write_source_data(output_dir / "gap_aware_scheme_change_source_data.csv", gap_rows, formula_x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
