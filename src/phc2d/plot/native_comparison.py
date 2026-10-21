"""Native model comparison figures from curated, provenance-bound CSV inputs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

from pathlib import Path


def render_native_comparison(csv_path: str | Path, output_stem: str | Path) -> list[Path]:
    """Render the historical two-panel layout without fitting or evaluating models."""
    import csv
    from pathlib import Path
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # ── Data ──────────────────────────────────────────────────────────────────────
    CSV_PATH = Path(csv_path)

    models = []
    k5_vals, k20_vals, f50_vals, mean_vals = [], [], [], []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(row["model"])
            k5_vals.append(float(row["K5"]))
            k20_vals.append(float(row["K20"]))
            f50_vals.append(float(row["F50"]))
            mean_vals.append(float(row["mean"]))

    n_models = len(models)
    x_pos = np.arange(n_models) * 1.12
    # Short, line-broken labels keep the comparison readable at slide scale.
    DISPLAY_NAMES = {
        "ResMLP-Wide": "ResMLP-\nWide",
        "Lifted CNN": "Lifted\nCNN",
        "MLP-Mixer": "MLP-\nMixer",
        "Gated ResMLP": "Gated\nResMLP",
        "TFNO-style v3": "TFNO-\nstyle",
        "Global–Local": "Global–\nLocal",
    }
    display_models = [DISPLAY_NAMES.get(name, name) for name in models]

    # Separate convolutional/MLP models from the operator families.
    network_models = {"MLP", "ResMLP-Wide", "Lifted CNN", "MLP-Mixer", "Gated ResMLP", "ConvNeXt"}
    NN_END = sum(name in network_models for name in models) - 1

    # ── Color palettes ────────────────────────────────────────────────────────────
    LEFT_COLORS = {
        "K5":  "#aeb5e8",
        "K20": "#b5f2a5",
        "F50": "#e9a5a0",
    }
    LEFT_EDGE = {
        "K5":  "#4255c8",
        "K20": "#4aa33b",
        "F50": "#c83c32",
    }

    RIGHT_COLOR = "#7254c4"
    RIGHT_EDGE = "#4b2e91"

    GL_COLOR = "#f01828"
    GL_EDGE = "#8e0b16"

    # Background shading colors
    BG_NN = "#e8f0fe"       # light blue for neural network models
    BG_NO = "#fef3e2"       # light amber for neural operator models

    # ── Font settings ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": ["Times New Roman", "SimSun"],
        "font.weight": "bold",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": 28,
        "ytick.labelsize": 32,
        "legend.fontsize": 27,
        "axes.labelsize": 32,
    })

    # ── Figure layout ─────────────────────────────────────────────────────────────
     # 33.87:10.76 aspect ratio for the target widescreen slide layout.
    fig = plt.figure(figsize=(40, 12.70))

    ax1 = fig.add_axes([0.04, 0.22, 0.59, 0.68])
    ax2 = fig.add_axes([0.69, 0.22, 0.27, 0.68])

    bar_width = 0.22
    gap = 0.04

    # ── Helper: add background shading ───────────────────────────────────────────
    def add_background(ax, x, nn_end):
        """Shade NN region (light blue) and NO region (light amber)."""
        half_step = (x[1] - x[0]) / 2
        # NN region: from before first model to after nn_end model
        ax.axvspan(x[0] - half_step - 0.15, x[nn_end] + half_step + 0.15,
                   color=BG_NN, alpha=0.55, zorder=0)
        # NO region: from after nn_end to after last model
        ax.axvspan(x[nn_end] + half_step + 0.15, x[-1] + half_step + 0.15,
                   color=BG_NO, alpha=0.55, zorder=0)


    # ── Helper: draw grouped bars ───────────────────────────────────────────────
    def draw_grouped_bars(ax, x, k5, k20, f50, colors, edges, y_label):
        n = len(x)
        total_width = 3 * bar_width + 2 * gap
        offset = -total_width / 2 + bar_width / 2
        offset2 = offset + bar_width + gap
        offset3 = offset2 + bar_width + gap

        def _fmt(v):
            return f"{v:.2f}" if v >= 1 else f"{v:.3f}"

        def _dy(val):
            """Small offset above bar; slightly larger for very short bars."""
            return 0.10 if val < 0.15 else 0.06

        # K5 bars
        for i in range(n):
            color = colors["K5"] if not (i == n - 1) else GL_COLOR
            edge = edges["K5"] if not (i == n - 1) else GL_EDGE
            ax.bar(x[i] + offset, k5[i], bar_width, color=color,
                   edgecolor=edge, linewidth=1.2, zorder=3)
            ax.text(x[i] + offset, k5[i] + _dy(k5[i]), _fmt(k5[i]),
                    ha="center", va="bottom", fontsize=22, fontweight="bold",
                    color="#32156e" if i != n - 1 else GL_COLOR, zorder=4)

        # K20 bars
        for i in range(n):
            color = colors["K20"] if not (i == n - 1) else GL_COLOR
            edge = edges["K20"] if not (i == n - 1) else GL_EDGE
            ax.bar(x[i] + offset2, k20[i], bar_width, color=color,
                   edgecolor=edge, linewidth=1.2, zorder=3)
            ax.text(x[i] + offset2, k20[i] + _dy(k20[i]), _fmt(k20[i]),
                    ha="center", va="bottom", fontsize=22, fontweight="bold",
                    color="#32156e" if i != n - 1 else GL_COLOR, zorder=4)

        # F50 bars
        for i in range(n):
            color = colors["F50"] if not (i == n - 1) else GL_COLOR
            edge = edges["F50"] if not (i == n - 1) else GL_EDGE
            ax.bar(x[i] + offset3, f50[i], bar_width, color=color,
                   edgecolor=edge, linewidth=1.2, zorder=3)
            ax.text(x[i] + offset3, f50[i] + _dy(f50[i]), _fmt(f50[i]),
                    ha="center", va="bottom", fontsize=22, fontweight="bold",
                    color="#32156e" if i != n - 1 else GL_COLOR, zorder=4)
            # Star for Global-Local
            if i == n - 1:
                ax.plot(x[i] + offset3, f50[i] + _dy(f50[i]) + 0.10,
                        marker="*", color=GL_COLOR,
                        markersize=16, markeredgecolor=GL_EDGE, markeredgewidth=1.5,
                        zorder=5, clip_on=False)

        ax.set_xticks(x)
        ax.set_xticklabels(display_models, rotation=0, ha="center", fontsize=28,
                           linespacing=0.95)
        ax.set_ylabel(y_label, fontsize=16, fontweight="bold")
        ax.yaxis.grid(True, color="#d0d7e2", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        legend_elements = [
            Rectangle((0, 0), 1, 1, fc=colors["K5"], ec=edges["K5"], label="K5"),
            Rectangle((0, 0), 1, 1, fc=colors["K20"], ec=edges["K20"], label="K20"),
            Rectangle((0, 0), 1, 1, fc=colors["F50"], ec=edges["F50"], label="F50"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=27,
                  frameon=False, ncol=3)


    # ─ Helper: draw mean bars (right chart) ────────────────────────────────────
    def draw_mean_bars(ax, x, means):
        for i, value in enumerate(means):
            color = GL_COLOR if i == len(means) - 1 else RIGHT_COLOR
            edge = GL_EDGE if i == len(means) - 1 else RIGHT_EDGE
            ax.bar(x[i], value, 0.62, color=color, edgecolor=edge,
                   linewidth=1.2, zorder=3)
            dy = 0.10 if value < 0.15 else 0.06
            ax.text(x[i], value + dy,
                    f"{value:.2f}" if value >= 1 else f"{value:.3f}",
                    ha="center", va="bottom", fontsize=22, fontweight="bold",
                    color=GL_COLOR if i == len(means) - 1 else "#32156e",
                    zorder=4)
        # Star for Global-Local
        last_dy = 0.10 if means[-1] < 0.15 else 0.06
        ax.plot(x[-1], means[-1] + last_dy + 0.12, marker="*", color=GL_COLOR,
                markersize=16, markeredgecolor=GL_EDGE, markeredgewidth=1.5,
                zorder=5, clip_on=False)
        ax.set_xticks(x)
        # The mean panel is narrower; a modest angle prevents long labels from colliding.
        ax.set_xticklabels(display_models, rotation=28, ha="right", fontsize=26,
                           linespacing=0.95)
        ax.set_ylabel("Mean MRE (%)", fontsize=16, fontweight="bold")
        ax.yaxis.grid(True, color="#d0d7e2", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    # ── Draw charts ───────────────────────────────────────────────────────────────
    # Background shading first (zorder=0)
    add_background(ax1, x_pos, NN_END)
    add_background(ax2, x_pos, NN_END)

    # Left: grouped bars
    draw_grouped_bars(
        ax1, x_pos, k5_vals, k20_vals, f50_vals,
        LEFT_COLORS, LEFT_EDGE,
        y_label="Bands 2–10 MRE (%)",
    )

    # Right: mean bars
    draw_mean_bars(ax2, x_pos, mean_vals)


    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg", "pdf"):
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=200 if extension == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
