"""Plot representative EXP-FWD-007 truth/prediction band diagrams.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


COLORS = {
    "truth": "#155FA0",
    "prediction": "#C83E3A",
    "gap": "#B8CAD8",
    "grid": "#D9D9D9",
    "text": "#303030",
}
SELECTION_LABELS = ("Low error", "Median error", "High error")
EVALUATIONS = {
    "Binary": {
        "single": ("shard_02", "sixdom_072.npz"),
        "joint": ("shard_06", "sixdom_096.npz"),
        "dataset": "full_cell_binary_fourier_k5_bz9_10000_xingyi_v1.h5",
    },
    "Graded": {
        "single": ("shard_09", "sixdom_079.npz"),
        "joint": ("shard_07", "sixdom_097.npz"),
        "dataset": "full_cell_graded_fourier_k5_bz9_10000_xingyi_v1.h5",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot K5 ConvNeXt and dual-lane predictions against true band diagrams."
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
    default = (
        root
        / "docs"
        / "results"
        / "forward"
        / "assets"
        / "exp_fwd_007_six_domain_20260831"
    )
    if requested is None:
        return default
    return requested if requested.is_absolute() else root / requested


def _high_symmetry_path(k_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    positive = float(np.max(k_points[:, 0]))
    grid = np.unique(k_points[:, 0])
    nonnegative = grid[grid >= -1e-12]
    coordinates = (
        [(float(value), 0.0) for value in nonnegative]
        + [(positive, float(value)) for value in nonnegative[1:]]
        + [(float(value), float(value)) for value in nonnegative[-2::-1]]
    )
    indices = []
    for coordinate in coordinates:
        distance = np.linalg.norm(k_points - np.asarray(coordinate), axis=1)
        index = int(np.argmin(distance))
        if float(distance[index]) > 1e-9:
            raise ValueError(f"Missing high-symmetry coordinate {coordinate}.")
        indices.append(index)
    segment = len(nonnegative) - 1
    ticks = np.asarray([0, segment, 2 * segment, 3 * segment], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64), ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"]


def _complete_gaps(target: np.ndarray) -> list[tuple[int, float, float]]:
    lower_edges = np.max(target, axis=0)[:-1]
    upper_edges = np.min(target, axis=0)[1:]
    return [
        (band + 1, float(lower_edges[band]), float(upper_edges[band]))
        for band in range(len(lower_edges))
        if upper_edges[band] > lower_edges[band]
    ]


def _select_gap_samples(target: np.ndarray, sample_mre: np.ndarray) -> list[int]:
    gap_mask = np.asarray([bool(_complete_gaps(sample)) for sample in target], dtype=bool)
    candidates = np.flatnonzero(gap_mask)
    if len(candidates) < 3:
        raise ValueError("At least three true complete-gap samples are required.")
    ordered = candidates[np.argsort(sample_mre[candidates], kind="stable")]
    return [int(ordered[0]), int(ordered[len(ordered) // 2]), int(ordered[-1])]


def _load_lane(root: Path, lane: str) -> dict[str, object]:
    spec = EVALUATIONS[lane]
    shard_root = (
        root
        / "runs"
        / "20260824"
        / "EXP-FWD-007_six_domain_20260824162756"
        / "shards"
    )
    single_path = shard_root / spec["single"][0] / spec["single"][1]
    joint_path = shard_root / spec["joint"][0] / spec["joint"][1]
    with np.load(single_path) as archive:
        single = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(joint_path) as archive:
        joint = {key: np.asarray(archive[key]) for key in archive.files}
    if not np.array_equal(single["target"], joint["target"]):
        raise ValueError(f"{lane} single-lane and joint targets are not aligned.")
    dataset_path = root / "data" / str(spec["dataset"])
    with h5py.File(dataset_path, "r") as handle:
        k_points = np.asarray(handle["k_points"], dtype=np.float64)
        idx_test = np.asarray(handle["idx_test"], dtype=np.int64)
    path_indices, ticks, tick_labels = _high_symmetry_path(k_points)
    selected = _select_gap_samples(joint["target"], joint["sample_mre"])
    return {
        "single": single,
        "joint": joint,
        "path_indices": path_indices,
        "ticks": ticks,
        "tick_labels": tick_labels,
        "idx_test": idx_test,
        "selected": selected,
        "gap_sample_count": sum(bool(_complete_gaps(sample)) for sample in joint["target"]),
    }


def _write_source_data(output_dir: Path, lane_data: dict[str, dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for lane, payload in lane_data.items():
        single = payload["single"]
        joint = payload["joint"]
        path_indices = payload["path_indices"]
        for selection, position in zip(SELECTION_LABELS, payload["selected"]):
            gaps = _complete_gaps(joint["target"][position])
            gap_text = ";".join(
                f"B{pair}-B{pair + 1}:{upper - lower:.12g}" for pair, lower, upper in gaps
            )
            for path_position, k_index in enumerate(path_indices):
                for band in range(joint["target"].shape[2]):
                    rows.append(
                        {
                            "lane": lane,
                            "selection": selection,
                            "evaluation_position": position,
                            "dataset_index": int(payload["idx_test"][position]),
                            "path_position": path_position,
                            "full_bz_k_index": int(k_index),
                            "band": band + 1,
                            "truth": f"{joint['target'][position, k_index, band]:.12g}",
                            "convnext_prediction": f"{single['prediction'][position, k_index, band]:.12g}",
                            "joint_prediction": f"{joint['prediction'][position, k_index, band]:.12g}",
                            "convnext_sample_mre_pct": f"{single['sample_mre'][position]:.12g}",
                            "joint_sample_mre_pct": f"{joint['sample_mre'][position]:.12g}",
                            "true_complete_gaps": gap_text,
                        }
                    )
    path = output_dir / "source_data_exp_fwd_007_band_diagrams.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(lane_data: dict[str, dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), sharex=True, sharey=True)
    panel_index = 0
    for row, (lane, payload) in enumerate(lane_data.items()):
        single = payload["single"]
        joint = payload["joint"]
        path_indices = payload["path_indices"]
        x = np.arange(len(path_indices))
        for column, (selection, position) in enumerate(zip(SELECTION_LABELS, payload["selected"])):
            ax = axes[row, column]
            target = joint["target"][position]
            for band in range(target.shape[1]):
                ax.plot(
                    x,
                    target[path_indices, band],
                    color=COLORS["truth"],
                    linewidth=1.05,
                    alpha=0.88,
                    zorder=3,
                )
                ax.plot(
                    x,
                    single["prediction"][position, path_indices, band],
                    color=COLORS["prediction"],
                    linewidth=0.72,
                    linestyle=(0, (3, 2)),
                    alpha=0.50,
                    zorder=2,
                )
                ax.plot(
                    x,
                    joint["prediction"][position, path_indices, band],
                    color=COLORS["prediction"],
                    linewidth=0.78,
                    alpha=0.92,
                    zorder=4,
                )
            gaps = _complete_gaps(target)
            for pair, lower, upper in gaps:
                ax.axhspan(lower, upper, color=COLORS["gap"], alpha=0.80, zorder=1)
                ax.text(
                    len(x) - 0.2,
                    (lower + upper) / 2,
                    f"B{pair}–B{pair + 1}",
                    ha="right",
                    va="center",
                    fontsize=4.8,
                    color=COLORS["text"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
                )
            single_mre = float(single["sample_mre"][position])
            joint_mre = float(joint["sample_mre"][position])
            delta = joint_mre - single_mre
            ax.set_title(
                f"{selection} | test {int(payload['idx_test'][position])}\n"
                f"ConvNeXt {single_mre:.2f}%  joint {joint_mre:.2f}%  Δ {delta:+.2f} pp",
                fontsize=6.4,
                color=COLORS["text"],
            )
            ax.set_xticks(payload["ticks"], payload["tick_labels"])
            ax.set_xlim(0, len(x) - 1)
            ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.55)
            ax.tick_params(labelsize=6, length=2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(
                -0.15,
                1.05,
                chr(ord("a") + panel_index),
                transform=ax.transAxes,
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            panel_index += 1
        axes[row, 0].set_ylabel(f"{lane}\n" + r"$\omega a/(2\pi c)$", fontsize=7)
        axes[row, 2].text(
            1.02,
            0.02,
            f"true-gap subset n={payload['gap_sample_count']}",
            transform=axes[row, 2].transAxes,
            rotation=90,
            ha="left",
            va="bottom",
            fontsize=5.4,
            color=COLORS["text"],
        )
    for ax in axes[-1, :]:
        ax.set_xlabel("k path", fontsize=7)
    handles = [
        Line2D([], [], color=COLORS["truth"], lw=1.2, label="True bands"),
        Line2D([], [], color=COLORS["prediction"], lw=0.9, linestyle=(0, (3, 2)), alpha=0.60, label="ConvNeXt prediction"),
        Line2D([], [], color=COLORS["prediction"], lw=1.0, label="Joint prediction"),
        Patch(facecolor=COLORS["gap"], edgecolor="none", alpha=0.80, label="True complete gap (full BZ)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.2)
    fig.suptitle(
        "EXP-FWD-007: predicted and true K5 band diagrams on complete-gap samples",
        fontsize=9.2,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.13, top=0.88, hspace=0.38, wspace=0.20)
    stem = output_dir / "exp_fwd_007_k5_gap_sample_band_diagrams"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = _output_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lane_data = {lane: _load_lane(root, lane) for lane in EVALUATIONS}
    _write_source_data(output_dir, lane_data)
    _plot(lane_data, output_dir)
    print(f"Created EXP-FWD-007 band diagrams in {output_dir}")


if __name__ == "__main__":
    main()
