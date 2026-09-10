"""Plot the EXP-FWD-007 six-domain surrogate evaluation landscape.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


COLORS = {
    "blue": "#155FA0",
    "blue_light": "#B9D7EA",
    "red": "#C83E3A",
    "red_light": "#F2C9C5",
    "gold": "#D9A526",
    "teal": "#2D8C8C",
    "purple": "#7A62A3",
    "gray": "#737373",
    "gray_light": "#E6E6E6",
    "grid": "#D9D9D9",
    "text": "#303030",
}

DOMAIN_ORDER = ("K5-B", "K5-G", "K20-B", "K20-G", "F50-B", "F50-G")
MODEL_ORDER = (
    "mlp",
    "res_mlp_wide",
    "lifted_cnn",
    "fno2d_band_v2",
    "mlp_mixer_band_v1",
    "res_mlp_gated_v2",
    "convnext_band_v1",
    "tfno2d_band_v3",
)
MODEL_LABELS = {
    "mlp": "MLP",
    "res_mlp_wide": "Res-MLP",
    "lifted_cnn": "Lifted CNN",
    "fno2d_band_v2": "FNO2D",
    "mlp_mixer_band_v1": "MLP-Mixer",
    "res_mlp_gated_v2": "Gated Res-MLP",
    "convnext_band_v1": "ConvNeXt",
    "tfno2d_band_v3": "TFNO2D",
    "dual_lane_band_operator_v1": "Dual-lane",
}
SHIFT_ORDER = (
    "native",
    "same_family_cross_lane",
    "cross_generator_same_lane",
    "generator_and_lane",
)
SHIFT_LABELS = {
    "native": "Native",
    "same_family_cross_lane": "Lane\nshift",
    "cross_generator_same_lane": "Generator\nshift",
    "generator_and_lane": "Generator +\nlane shift",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready figures from EXP-FWD-007 evaluation cells."
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
        return (
            root
            / "docs"
            / "results"
            / "forward"
            / "assets"
            / "exp_fwd_007_six_domain_20260831"
        )
    return requested if requested.is_absolute() else root / requested


def _read_cells(root: Path) -> list[dict[str, str]]:
    path = (
        root
        / "runs"
        / "20260824"
        / "EXP-FWD-007_six_domain_20260824162756"
        / "evaluation_cells.csv"
    )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 510:
        raise ValueError(f"Expected 510 evaluation cells, found {len(rows)}.")
    if any(row["status"] != "completed" for row in rows):
        raise ValueError("The evaluation table contains non-completed cells.")
    return rows


def _metric(row: dict[str, str]) -> float:
    return float(row["bands_2_10_mre_pct"])


def _source_domain(row: dict[str, str]) -> str:
    return f"{row['source_family']}-{row['source_lane']}"


def _seed0_single_lane(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["source_seed"] == "0" and row["source_lane"] in {"B", "G"}
    ]


def _matrix_summary(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, list[list[str]], np.ndarray]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in _seed0_single_lane(rows):
        grouped[(_source_domain(row), row["target_domain"])].append(row)

    mean = np.empty((len(DOMAIN_ORDER), len(DOMAIN_ORDER)), dtype=float)
    best = np.empty_like(mean)
    winner: list[list[str]] = [[""] * len(DOMAIN_ORDER) for _ in DOMAIN_ORDER]
    counts = np.empty_like(mean, dtype=int)
    for i, source in enumerate(DOMAIN_ORDER):
        for j, target in enumerate(DOMAIN_ORDER):
            cell = grouped[(source, target)]
            if len(cell) != len(MODEL_ORDER):
                raise ValueError(f"Expected 8 models for {source} -> {target}, found {len(cell)}.")
            values = np.asarray([_metric(row) for row in cell], dtype=float)
            index = int(np.argmin(values))
            mean[i, j] = float(np.mean(values))
            best[i, j] = float(values[index])
            winner[i][j] = MODEL_LABELS[cell[index]["source_model"]]
            counts[i, j] = len(cell)
    return mean, best, winner, counts


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_source_data(rows: list[dict[str, str]], output_dir: Path) -> None:
    mean, best, winner, counts = _matrix_summary(rows)
    matrix_rows: list[dict[str, object]] = []
    for i, source in enumerate(DOMAIN_ORDER):
        for j, target in enumerate(DOMAIN_ORDER):
            matrix_rows.append(
                {
                    "source_domain": source,
                    "target_domain": target,
                    "model_count": int(counts[i, j]),
                    "mean_bands_2_10_mre_pct": f"{mean[i, j]:.8f}",
                    "best_bands_2_10_mre_pct": f"{best[i, j]:.8f}",
                    "best_model": winner[i][j],
                }
            )
    _write_csv(
        output_dir / "source_data_six_domain_matrix.csv",
        [
            "source_domain",
            "target_domain",
            "model_count",
            "mean_bands_2_10_mre_pct",
            "best_bands_2_10_mre_pct",
            "best_model",
        ],
        matrix_rows,
    )

    k5_rows = [row for row in rows if row["source_family"] == "K5"]
    _write_csv(
        output_dir / "source_data_k5_source_models.csv",
        [
            "source_model",
            "source_lane",
            "source_seed",
            "target_domain",
            "comparison_kind",
            "bands_2_10_mre_pct",
            "sample_count",
        ],
        [
            {
                key: row[key]
                for key in (
                    "source_model",
                    "source_lane",
                    "source_seed",
                    "target_domain",
                    "comparison_kind",
                    "bands_2_10_mre_pct",
                    "sample_count",
                )
            }
            for row in k5_rows
        ],
    )

    shift_rows = _seed0_single_lane(rows)
    _write_csv(
        output_dir / "source_data_seed0_shift_cells.csv",
        [
            "source_family",
            "source_model",
            "source_lane",
            "target_domain",
            "comparison_kind",
            "bands_2_10_mre_pct",
        ],
        [
            {
                key: row[key]
                for key in (
                    "source_family",
                    "source_model",
                    "source_lane",
                    "target_domain",
                    "comparison_kind",
                    "bands_2_10_mre_pct",
                )
            }
            for row in shift_rows
        ],
    )


def _metric_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "mre",
        ["#F7FBFF", "#9ECAE1", "#FDD0A2", "#EF6548", "#8C2D04"],
    )


def _annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    fmt: str = ".2f",
    fontsize: float = 5.4,
    second_lines: list[list[str]] | None = None,
) -> None:
    norm = LogNorm(vmin=0.25, vmax=16.0)
    cmap = _metric_cmap()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.53 else COLORS["text"]
            text = format(value, fmt)
            if second_lines is not None:
                text += "\n" + second_lines[i][j]
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=color)


def _style_matrix_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xticks(range(len(DOMAIN_ORDER)), DOMAIN_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(DOMAIN_ORDER)), DOMAIN_ORDER)
    ax.set_xlabel("Target domain")
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
    ):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _plot_transfer_atlas(rows: list[dict[str, str]], output_dir: Path) -> None:
    mean, best, winner, _ = _matrix_summary(rows)
    cmap = _metric_cmap()
    norm = LogNorm(vmin=0.25, vmax=16.0)

    fig = plt.figure(figsize=(7.2, 6.3))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.12, 0.88),
        width_ratios=(1.0, 1.0, 0.045),
        hspace=0.48,
        wspace=0.42,
    )
    ax_mean = fig.add_subplot(grid[0, 0])
    ax_best = fig.add_subplot(grid[0, 1])
    ax_cbar = fig.add_subplot(grid[0, 2])
    ax_shift = fig.add_subplot(grid[1, :2])

    im = ax_mean.imshow(mean, cmap=cmap, norm=norm, aspect="equal")
    _annotate_heatmap(ax_mean, mean)
    _style_matrix_axes(ax_mean, "Source domain")
    ax_mean.set_title("Mean across 8 single-lane models (seed 0)", fontsize=8)
    _add_panel_label(ax_mean, "a")

    ax_best.imshow(best, cmap=cmap, norm=norm, aspect="equal")
    _annotate_heatmap(ax_best, best, fontsize=4.5, second_lines=winner)
    _style_matrix_axes(ax_best, "Source domain")
    ax_best.set_title("Best available model in each transfer cell", fontsize=8)
    _add_panel_label(ax_best, "b")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Bands 2–10 MRE (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    values_by_shift = {
        kind: np.asarray(
            [_metric(row) for row in _seed0_single_lane(rows) if row["comparison_kind"] == kind],
            dtype=float,
        )
        for kind in SHIFT_ORDER
    }
    positions = np.arange(len(SHIFT_ORDER), dtype=float)
    box = ax_shift.boxplot(
        [values_by_shift[kind] for kind in SHIFT_ORDER],
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["text"], "linewidth": 1.2},
        whiskerprops={"color": COLORS["gray"], "linewidth": 0.8},
        capprops={"color": COLORS["gray"], "linewidth": 0.8},
    )
    box_colors = (COLORS["blue_light"], "#D7DCE2", "#EDC2AE", COLORS["red_light"])
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(COLORS["gray"])
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(7)
    for x, kind, color in zip(
        positions,
        SHIFT_ORDER,
        (COLORS["blue"], COLORS["gray"], COLORS["gold"], COLORS["red"]),
    ):
        values = values_by_shift[kind]
        jitter = rng.uniform(-0.19, 0.19, size=len(values))
        ax_shift.scatter(
            np.full_like(values, x) + jitter,
            values,
            s=7,
            color=color,
            alpha=0.42,
            linewidths=0,
            rasterized=False,
        )
        mean_value = float(np.mean(values))
        ax_shift.scatter([x], [mean_value], marker="D", s=29, color="white", edgecolor=color, zorder=4)
        ax_shift.text(
            x,
            mean_value * 1.18,
            f"mean {mean_value:.2f}\nn={len(values)}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=COLORS["text"],
        )
    ax_shift.set_yscale("log")
    ax_shift.set_ylim(0.2, 55)
    ax_shift.set_xticks(positions, [SHIFT_LABELS[kind] for kind in SHIFT_ORDER])
    ax_shift.set_ylabel("Bands 2–10 MRE (%)")
    ax_shift.set_title("Domain shift increases error and broadens model-to-model spread", fontsize=8)
    ax_shift.grid(axis="y", which="both", color=COLORS["grid"], linewidth=0.55, alpha=0.7)
    ax_shift.spines[["top", "right"]].set_visible(False)
    _add_panel_label(ax_shift, "c")
    ax_shift.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=COLORS["gray"], alpha=0.5, label="one evaluation cell"),
            Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", markeredgecolor=COLORS["text"], label="category mean"),
        ],
        loc="upper left",
        fontsize=6,
        frameon=False,
    )

    fig.suptitle(
        "EXP-FWD-007: six-domain surrogate transfer atlas",
        x=0.5,
        y=0.995,
        fontsize=10,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.11, top=0.92)
    _save_figure(fig, output_dir, "exp_fwd_007_six_domain_transfer_atlas")


def _k5_matrix(rows: list[dict[str, str]], lane: str) -> np.ndarray:
    matrix = np.empty((len(MODEL_ORDER), len(DOMAIN_ORDER)), dtype=float)
    for i, model in enumerate(MODEL_ORDER):
        for j, target in enumerate(DOMAIN_ORDER):
            matches = [
                row
                for row in rows
                if row["source_family"] == "K5"
                and row["source_lane"] == lane
                and row["source_model"] == model
                and row["target_domain"] == target
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one K5-{lane} {model} -> {target} cell.")
            matrix[i, j] = _metric(matches[0])
    return matrix


def _plot_k5_landscape(rows: list[dict[str, str]], output_dir: Path) -> None:
    cmap = _metric_cmap()
    norm = LogNorm(vmin=0.25, vmax=16.0)
    binary = _k5_matrix(rows, "B")
    graded = _k5_matrix(rows, "G")

    joint = np.asarray(
        [
            _metric(
                next(
                    row
                    for row in rows
                    if row["source_family"] == "K5"
                    and row["source_lane"] == "J"
                    and row["target_domain"] == target
                )
            )
            for target in DOMAIN_ORDER
        ],
        dtype=float,
    )[None, :]

    fig = plt.figure(figsize=(7.2, 6.5))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.19),
        width_ratios=(1.0, 1.0, 0.045),
        hspace=0.45,
        wspace=0.27,
    )
    ax_b = fig.add_subplot(grid[0, 0])
    ax_g = fig.add_subplot(grid[0, 1])
    ax_joint = fig.add_subplot(grid[1, :2])
    ax_cbar = fig.add_subplot(grid[:, 2])

    for ax, matrix, lane, label in (
        (ax_b, binary, "Binary", "a"),
        (ax_g, graded, "Graded", "b"),
    ):
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        _annotate_heatmap(ax, matrix, fontsize=5.2)
        ax.set_xticks(range(len(DOMAIN_ORDER)), DOMAIN_ORDER, rotation=35, ha="right")
        ax.set_yticks(range(len(MODEL_ORDER)))
        if ax is ax_b:
            ax.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
            ax.set_ylabel("K5 source model")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Target domain")
        ax.set_title(f"K5 {lane}-source checkpoints (seed 0)", fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        native_col = 0 if lane == "Binary" else 1
        ax.add_patch(
            plt.Rectangle(
                (native_col - 0.49, -0.49),
                0.98,
                len(MODEL_ORDER) - 0.02,
                fill=False,
                edgecolor=COLORS["blue"],
                linewidth=1.5,
            )
        )
        _add_panel_label(ax, label)

    ax_joint.imshow(joint, cmap=cmap, norm=norm, aspect="auto")
    _annotate_heatmap(ax_joint, joint, fontsize=6.2)
    ax_joint.set_xticks(range(len(DOMAIN_ORDER)), DOMAIN_ORDER)
    ax_joint.set_yticks([0], ["Dual-lane"])
    ax_joint.set_xlabel("Target domain")
    ax_joint.set_title("K5 joint checkpoint, reported separately", fontsize=8)
    ax_joint.tick_params(length=0)
    for spine in ax_joint.spines.values():
        spine.set_visible(False)
    _add_panel_label(ax_joint, "c")

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Bands 2–10 MRE (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle(
        "EXP-FWD-007: K5 model performance changes with the target domain",
        x=0.5,
        y=0.995,
        fontsize=10,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.10, top=0.92)
    _save_figure(fig, output_dir, "exp_fwd_007_k5_model_landscape")


def _validate(rows: list[dict[str, str]], output_dir: Path) -> None:
    seed0 = [row for row in rows if row["source_seed"] == "0"]
    seed_counts = Counter(row["source_seed"] for row in rows)
    if seed_counts != Counter({"0": 306, "1": 102, "2": 102}):
        raise ValueError(f"Unexpected seed distribution: {seed_counts}")
    comparison_counts = Counter(
        row["comparison_kind"] for row in _seed0_single_lane(rows)
    )
    expected = Counter(
        {
            "native": 48,
            "same_family_cross_lane": 48,
            "cross_generator_same_lane": 96,
            "generator_and_lane": 96,
        }
    )
    if comparison_counts != expected:
        raise ValueError(f"Unexpected seed-0 comparison counts: {comparison_counts}")
    if len(seed0) != 306:
        raise ValueError("Seed-0 common-source slice must contain 306 cells including joint.")
    for stem in (
        "exp_fwd_007_six_domain_transfer_atlas",
        "exp_fwd_007_k5_model_landscape",
    ):
        for suffix in ("svg", "pdf", "png"):
            path = output_dir / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty export: {path}")
    for name in (
        "source_data_six_domain_matrix.csv",
        "source_data_k5_source_models.csv",
        "source_data_seed0_shift_cells.csv",
    ):
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing source-data export: {path}")


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = _output_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_cells(root)
    _export_source_data(rows, output_dir)
    _plot_transfer_atlas(rows, output_dir)
    _plot_k5_landscape(rows, output_dir)
    _validate(rows, output_dir)
    print(f"Created EXP-FWD-007 figure bundle in {output_dir}")


if __name__ == "__main__":
    main()
