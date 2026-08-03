"""Generate the manuscript summary figure from traceable CSV inputs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

BLUE = "#2F6B9A"
ORANGE = "#D9792B"
GREEN = "#4D8B66"
GRAY = "#7A7A7A"
LIGHT = "#DCE8F2"


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    return float(value) if value else float("nan")


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def clean_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8, zorder=0)
    ax.tick_params(length=3)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def main() -> None:
    style()
    forward = read_rows("forward_results.csv")
    query = read_rows("query_results.csv")
    inverse = read_rows("inverse_results.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.55), constrained_layout=True)

    ax = axes[0, 0]
    names = [row["method"] for row in forward]
    means = np.array([as_float(row["mre_mean_pct"]) for row in forward])
    stds = np.array([as_float(row["mre_std_pct"]) for row in forward])
    xpos = np.arange(len(names))
    colors = [BLUE, GRAY, GRAY, GRAY]
    ax.bar(xpos, means, yerr=stds, capsize=2.5, color=colors, width=0.68, edgecolor="white", zorder=3)
    for x, y in zip(xpos, means):
        ax.text(x, y + 0.10, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xpos, names, rotation=18, ha="right")
    ax.set_ylabel("Bands 2–10 test MRE (%)")
    ax.set_ylim(0, 2.85)
    ax.set_title("Full-band forward prediction (3 seeds)", loc="left")
    clean_axis(ax)
    panel_label(ax, "a")

    ax = axes[0, 1]
    protocols = ["All-k", "Interpolation", "Held segment"]
    methods = ["FNO query", "DeepONet", "MLP query"]
    method_colors = [BLUE, ORANGE, GREEN]
    x = np.arange(len(protocols))
    width = 0.23
    for idx, (method, color) in enumerate(zip(methods, method_colors)):
        vals = [as_float(next(row["mre_mean_pct"] for row in query if row["method"] == method and row["protocol"] == protocol)) for protocol in protocols]
        ax.bar(x + (idx - 1) * width, vals, width=width, label=method, color=color, edgecolor="white", zorder=3)
    ax.axvspan(1.55, 2.45, color="#F3E5D8", alpha=0.7, zorder=0)
    ax.set_xticks(x, protocols)
    ax.set_ylabel("Query MRE (%)")
    ax.set_ylim(0, 4.8)
    ax.set_title("Interpolation versus extrapolation", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2, columnspacing=0.8)
    clean_axis(ax)
    panel_label(ax, "b")

    ax = axes[1, 0]
    inv_names = [row["method"] for row in inverse]
    mpb = np.array([as_float(row["mpb_mre_pct"]) for row in inverse])
    low = np.array([as_float(row["mpb_ci_low_pct"]) for row in inverse])
    high = np.array([as_float(row["mpb_ci_high_pct"]) for row in inverse])
    yerr = np.vstack([mpb - low, high - mpb])
    colors = [BLUE, ORANGE]
    ax.bar(np.arange(2), mpb, yerr=yerr, capsize=3, color=colors, width=0.58, edgecolor="white", zorder=3)
    for x0, y0 in enumerate(mpb):
        ax.text(x0, y0 + 0.07, f"{y0:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(2), ["Single mode\n(no experts)", "Four modes"])
    ax.set_ylabel("Best-of-Top-5 MPB MRE (%)")
    ax.set_ylim(0, 2.75)
    ax.set_title("Solver-verified inverse design (seed 0)", loc="left")
    clean_axis(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    surrogate = np.array([as_float(row["surrogate_mre_pct"]) for row in inverse])
    gap = np.array([as_float(row["gap_pp"]) for row in inverse])
    ax.bar(np.arange(2), surrogate, color=LIGHT, edgecolor=BLUE, linewidth=0.9, width=0.58, label="BISON MRE", zorder=3)
    ax.bar(np.arange(2), gap, bottom=surrogate, color=[BLUE, ORANGE], edgecolor="white", width=0.58, label="Replay gap", zorder=3)
    for x0, s0, g0 in zip(np.arange(2), surrogate, gap):
        ax.text(x0, s0 + g0 + 0.06, f"+{g0:.3f} pp", ha="center", fontsize=7)
    ax.set_xticks(np.arange(2), ["Single mode", "Four modes"])
    ax.set_ylabel("MRE (%)")
    ax.set_ylim(0, 2.75)
    ax.set_title("Surrogate and replay evidence", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2)
    clean_axis(ax)
    panel_label(ax, "d")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "results_summary.pdf")
    fig.savefig(FIGURES / "results_summary.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
