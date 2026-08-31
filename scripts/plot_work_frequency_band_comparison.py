"""Plot surrogate bands and the work-frequency prior against MPB references.

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
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


PALETTE = {
    "truth": "#272727",
    "prediction": "#D9782D",
    "prior": "#6F5AA8",
    "midpoint": "#2F7F8F",
    "overlap": "#EBC6C2",
    "grid": "#D9D9D9",
}

PHYSICAL_SAMPLES = (
    ("binary", 89),
    ("binary", 96),
    ("graded", 62),
    ("graded", 337),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot surrogate/MPB band overlays and the scalar Bragg prior."
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


def _resolve_output_dir(root: Path, requested: Path | None) -> Path:
    if requested is None:
        return root / "docs" / "reports" / "assets" / "work_frequency_band_comparison_20260831"
    return requested if requested.is_absolute() else root / requested


def _high_symmetry_path(k_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return discrete Gamma-X-M-Gamma indices available on the frozen 9x9 grid."""
    positive = float(np.max(k_points[:, 0]))
    grid = np.unique(k_points[:, 0])
    nonnegative = grid[grid >= -1e-12]
    gamma_x = [(float(value), 0.0) for value in nonnegative]
    x_m = [(positive, float(value)) for value in nonnegative[1:]]
    m_gamma = [(float(value), float(value)) for value in nonnegative[-2::-1]]
    coordinates = gamma_x + x_m + m_gamma
    indices: list[int] = []
    for coordinate in coordinates:
        distance = np.linalg.norm(k_points - np.asarray(coordinate), axis=1)
        index = int(np.argmin(distance))
        if float(distance[index]) > 1e-9:
            raise ValueError(f"Missing high-symmetry coordinate {coordinate} in k grid.")
        indices.append(index)
    segment = len(nonnegative) - 1
    ticks = np.asarray([0, segment, 2 * segment, 3 * segment], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64), ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"]


def _load_prediction_panel(archive: Path) -> dict[str, object]:
    with np.load(archive, allow_pickle=False) as payload:
        target = np.asarray(payload["target_Wh"], dtype=np.float64)
        predictions = np.stack(
            [np.asarray(payload[f"prediction_seed{seed}"], dtype=np.float64) for seed in range(3)],
            axis=0,
        )
        prediction = np.mean(predictions, axis=0)
        sample_mre = np.mean(
            np.abs(prediction - target) / np.maximum(np.abs(target), 1e-8),
            axis=(1, 2),
        )
        order = np.argsort(sample_mre)
        position = int(order[len(order) // 2])
        return {
            "dataset_index": int(payload["indices"][position]),
            "sample_id": str(payload["sample_id"][position]),
            "target": target[position],
            "prediction": prediction[position],
            "mre_pct": float(100.0 * sample_mre[position]),
        }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _formula_frequencies(root: Path) -> dict[tuple[str, int], float]:
    rows = _read_csv_rows(
        root / "runs" / "20260828" / "EXP-PHY-005_202608281751" / "gap_alignment_rows.csv"
    )
    selected: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["lane"], int(row["dataset_index"]))
        if row["split"] == "validation" and row["candidate"] == "x" and key in PHYSICAL_SAMPLES:
            selected[key] = float(row["new_frequency"])
    missing = set(PHYSICAL_SAMPLES) - set(selected)
    if missing:
        raise ValueError(f"Missing x-prior rows for {sorted(missing)}.")
    return selected


def _refined_b1_b2(root: Path) -> dict[tuple[str, int], dict[str, float | str]]:
    rows = _read_csv_rows(
        root / "runs" / "20260829" / "EXP-PHY-006_202608291923" / "gap_refinement_samples.csv"
    )
    selected: dict[tuple[str, int], dict[str, float | str]] = {}
    for row in rows:
        if int(row["lower_band_one_based"]) != 1:
            continue
        key = (row["lane"], int(row["dataset_index"]))
        if key not in PHYSICAL_SAMPLES:
            continue
        selected[key] = {
            "lower_edge": float(row["refined_lower_edge_frequency"]),
            "upper_edge": float(row["refined_upper_edge_frequency"]),
            "midpoint": float(row["refined_center"]),
            "signed_width": float(row["refined_signed_width"]),
            "status": row["refined_status"],
        }
    missing = set(PHYSICAL_SAMPLES) - set(selected)
    if missing:
        raise ValueError(f"Missing refined B1-B2 rows for {sorted(missing)}.")
    return selected


def _dataset_path(root: Path, lane: str) -> Path:
    stem = "binary" if lane == "binary" else "graded"
    return root / "data" / f"full_cell_{stem}_fourier_k5_bz9_10000_xingyi_v1.h5"


def _style_axis(axis: plt.Axes, *, ticks: np.ndarray, labels: list[str]) -> None:
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_xlim(float(ticks[0]), float(ticks[-1]))
    axis.set_xlabel("Wave vector")
    axis.set_ylabel(r"Normalized frequency $\omega a/(2\pi c)$")
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.45, alpha=0.55)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.tick_params(width=0.7, length=3)
    for tick in ticks[1:-1]:
        axis.axvline(float(tick), color=PALETTE["grid"], linewidth=0.7, zorder=0)


def _plot_band_set(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    label: str,
) -> None:
    x = np.arange(values.shape[0])
    for band in range(values.shape[1]):
        axis.plot(
            x,
            values[:, band],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if band == 0 else None,
            zorder=3,
        )


def _write_source_data(
    output: Path,
    prediction_panels: dict[str, dict[str, object]],
    path_indices: np.ndarray,
    path_coordinates: np.ndarray,
    physical_bands: dict[tuple[str, int], np.ndarray],
    priors: dict[tuple[str, int], float],
    refined: dict[tuple[str, int], dict[str, float | str]],
) -> None:
    fields = [
        "panel_type",
        "lane",
        "dataset_index",
        "path_position",
        "kx",
        "ky",
        "band",
        "mpb_frequency",
        "surrogate_frequency",
        "formula_x_frequency",
        "refined_b1_b2_midpoint",
        "refined_b1_b2_signed_width",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for lane, panel in prediction_panels.items():
            target = np.asarray(panel["target"])[path_indices]
            prediction = np.asarray(panel["prediction"])[path_indices]
            for path_position, (kx, ky) in enumerate(path_coordinates):
                for band in range(target.shape[1]):
                    writer.writerow(
                        {
                            "panel_type": "surrogate_test_overlay",
                            "lane": lane,
                            "dataset_index": panel["dataset_index"],
                            "path_position": path_position,
                            "kx": f"{kx:.12g}",
                            "ky": f"{ky:.12g}",
                            "band": band + 1,
                            "mpb_frequency": f"{target[path_position, band]:.12g}",
                            "surrogate_frequency": f"{prediction[path_position, band]:.12g}",
                        }
                    )
        for key in PHYSICAL_SAMPLES:
            values = physical_bands[key][path_indices]
            detail = refined[key]
            for path_position, (kx, ky) in enumerate(path_coordinates):
                for band in range(values.shape[1]):
                    writer.writerow(
                        {
                            "panel_type": "formula_prior_physical_validation",
                            "lane": key[0],
                            "dataset_index": key[1],
                            "path_position": path_position,
                            "kx": f"{kx:.12g}",
                            "ky": f"{ky:.12g}",
                            "band": band + 1,
                            "mpb_frequency": f"{values[path_position, band]:.12g}",
                            "formula_x_frequency": f"{priors[key]:.12g}",
                            "refined_b1_b2_midpoint": f"{float(detail['midpoint']):.12g}",
                            "refined_b1_b2_signed_width": f"{float(detail['signed_width']):.12g}",
                        }
                    )


def _strip_svg_trailing_whitespace(path: Path) -> None:
    """Keep generated vector output compatible with repository diff checks."""
    content = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
    path.write_text(cleaned, encoding="utf-8", newline="\n")


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = _resolve_output_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_archives = {
        "binary": root
        / "runs"
        / "20260806"
        / "full_cell_forward_final_test_20260806"
        / "predictions_test_binary_binary.npz",
        "graded": root
        / "runs"
        / "20260806"
        / "full_cell_forward_final_test_20260806"
        / "predictions_test_graded_continuous.npz",
    }
    prediction_panels = {
        lane: _load_prediction_panel(path) for lane, path in prediction_archives.items()
    }
    priors = _formula_frequencies(root)
    refined = _refined_b1_b2(root)

    physical_bands: dict[tuple[str, int], np.ndarray] = {}
    k_points: np.ndarray | None = None
    for lane in ("binary", "graded"):
        with h5py.File(_dataset_path(root, lane), "r") as handle:
            lane_k_points = np.asarray(handle["k_points"], dtype=np.float64)
            if k_points is None:
                k_points = lane_k_points
            elif not np.allclose(k_points, lane_k_points):
                raise ValueError("Binary and Graded datasets do not share the same k grid.")
            for key in PHYSICAL_SAMPLES:
                if key[0] == lane:
                    physical_bands[key] = np.asarray(handle["Wh"][key[1]], dtype=np.float64)
    assert k_points is not None
    path_indices, ticks, tick_labels = _high_symmetry_path(k_points)
    path_coordinates = k_points[path_indices]

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 6.2,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(7.20, 7.05), constrained_layout=True)
    panel_labels = iter("abcdef")

    for column, lane in enumerate(("binary", "graded")):
        axis = axes[0, column]
        panel = prediction_panels[lane]
        target = np.asarray(panel["target"])[path_indices]
        prediction = np.asarray(panel["prediction"])[path_indices]
        _plot_band_set(
            axis,
            target,
            color=PALETTE["truth"],
            linestyle="-",
            linewidth=1.0,
            alpha=0.78,
            label="MPB target",
        )
        _plot_band_set(
            axis,
            prediction,
            color=PALETTE["prediction"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.82,
            label="Surrogate prediction",
        )
        _style_axis(axis, ticks=ticks, labels=tick_labels)
        axis.set_title(
            f"{lane.title()} held-out test | index {panel['dataset_index']} | "
            f"MRE {float(panel['mre_pct']):.2f}%",
            loc="left",
        )
        axis.legend(loc="upper left", ncol=2, handlelength=2.6, columnspacing=0.9)
        axis.text(
            -0.13,
            1.04,
            next(panel_labels),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    for position, key in enumerate(PHYSICAL_SAMPLES):
        row = 1 + position // 2
        column = position % 2
        axis = axes[row, column]
        values = physical_bands[key][path_indices]
        detail = refined[key]
        prior = priors[key]
        midpoint = float(detail["midpoint"])
        lower_edge = float(detail["lower_edge"])
        upper_edge = float(detail["upper_edge"])
        relative_error = 100.0 * abs(prior - midpoint) / abs(midpoint)
        _plot_band_set(
            axis,
            values,
            color=PALETTE["truth"],
            linestyle="-",
            linewidth=0.9,
            alpha=0.72,
            label="MPB bands (9x9 grid)",
        )
        axis.axhspan(
            min(lower_edge, upper_edge),
            max(lower_edge, upper_edge),
            color=PALETTE["overlap"],
            alpha=0.42,
            linewidth=0,
            label="Locally refined B1-B2 overlap",
            zorder=1,
        )
        axis.axhline(
            midpoint,
            color=PALETTE["midpoint"],
            linewidth=1.1,
            linestyle="-.",
            label="Refined B1-B2 midpoint",
            zorder=4,
        )
        axis.axhline(
            prior,
            color=PALETTE["prior"],
            linewidth=1.25,
            linestyle=":",
            label=r"Formula $x=1/(2n_{\mathrm{eff}})$",
            zorder=5,
        )
        _style_axis(axis, ticks=ticks, labels=tick_labels)
        axis.set_title(
            f"{key[0].title()} validation | index {key[1]} | prior error {relative_error:.2f}%",
            loc="left",
        )
        axis.text(
            0.985,
            0.035,
            f"B1-B2 signed width\n{float(detail['signed_width']):.4f} (pseudo-gap)",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.1,
            color="#7B3F3A",
        )
        axis.text(
            -0.13,
            1.04,
            next(panel_labels),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        if position == 0:
            axis.legend(loc="upper left", handlelength=2.8, fontsize=5.8)

    figure.suptitle(
        "Band predictions and MPB references: a scalar Bragg prior locates a region, not a band structure",
        fontsize=8.4,
        fontweight="bold",
    )
    base = output_dir / "work_frequency_band_prediction_vs_truth"
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    _strip_svg_trailing_whitespace(base.with_suffix(".svg"))
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    _write_source_data(
        output_dir / "work_frequency_band_prediction_vs_truth_source_data.csv",
        prediction_panels,
        path_indices,
        path_coordinates,
        physical_bands,
        priors,
        refined,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
