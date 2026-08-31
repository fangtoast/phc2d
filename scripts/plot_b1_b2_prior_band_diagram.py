"""Plot why the analytic x prior maps to the B1-B2 frequency region.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


COLORS = {
    "b1": "#155FA0",
    "b2": "#4C9AC6",
    "higher": "#9CB9CE",
    "prior": "#C83E3A",
    "midpoint": "#174A73",
    "overlap": "#D97C78",
    "grid": "#D9D9D9",
    "text": "#333333",
}

SAMPLES = (
    ("binary", 89),
    ("binary", 96),
    ("graded", 62),
    ("graded", 337),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the analytic x prior on real-MPB B1-B10 band diagrams."
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
        return root / "docs" / "reports" / "assets" / "b1_b2_prior_band_diagram_20260831"
    return requested if requested.is_absolute() else root / requested


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _high_symmetry_path(k_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract the available discrete Gamma-X-M-Gamma path from the 9x9 full-BZ grid."""
    positive = float(np.max(k_points[:, 0]))
    grid = np.unique(k_points[:, 0])
    nonnegative = grid[grid >= -1e-12]
    coordinates = (
        [(float(value), 0.0) for value in nonnegative]
        + [(positive, float(value)) for value in nonnegative[1:]]
        + [(float(value), float(value)) for value in nonnegative[-2::-1]]
    )
    indices: list[int] = []
    for coordinate in coordinates:
        distance = np.linalg.norm(k_points - np.asarray(coordinate), axis=1)
        index = int(np.argmin(distance))
        if float(distance[index]) > 1e-9:
            raise ValueError(f"Missing high-symmetry coordinate {coordinate}.")
        indices.append(index)
    segment = len(nonnegative) - 1
    ticks = np.asarray([0, segment, 2 * segment, 3 * segment], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64), ticks, [r"$\Gamma$", "X", "M", r"$\Gamma$"]


def _dataset_path(root: Path, lane: str) -> Path:
    return root / "data" / f"full_cell_{lane}_fourier_k5_bz9_10000_xingyi_v1.h5"


def _load_sample_data(
    root: Path,
) -> tuple[
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int], np.ndarray],
    dict[tuple[str, int], str],
    np.ndarray,
]:
    band_values: dict[tuple[str, int], np.ndarray] = {}
    structures: dict[tuple[str, int], np.ndarray] = {}
    structure_fields: dict[tuple[str, int], str] = {}
    shared_k_points: np.ndarray | None = None
    for lane in ("binary", "graded"):
        grid_field = "material_area_fraction_grid_64" if lane == "binary" else "material_weight_grid_64"
        with h5py.File(_dataset_path(root, lane), "r") as handle:
            k_points = np.asarray(handle["k_points"], dtype=np.float64)
            if shared_k_points is None:
                shared_k_points = k_points
            elif not np.allclose(shared_k_points, k_points):
                raise ValueError("Binary and Graded datasets do not share the frozen k grid.")
            for key in SAMPLES:
                if key[0] == lane:
                    band_values[key] = np.asarray(handle["Wh"][key[1]], dtype=np.float64)
                    structures[key] = np.asarray(handle[grid_field][key[1]], dtype=np.float64)
                    structure_fields[key] = grid_field
    assert shared_k_points is not None
    return band_values, structures, structure_fields, shared_k_points


def _load_formula_x(root: Path) -> dict[tuple[str, int], float]:
    rows = _read_csv(
        root / "runs" / "20260828" / "EXP-PHY-005_202608281751" / "gap_alignment_rows.csv"
    )
    selected: dict[tuple[str, int], float] = {}
    for key in SAMPLES:
        matches = [
            row
            for row in rows
            if row["split"] == "validation"
            and row["candidate"] == "x"
            and row["lane"] == key[0]
            and int(row["dataset_index"]) == key[1]
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one x-prior record for {key}, found {len(matches)}.")
        selected[key] = float(matches[0]["new_frequency"])
    return selected


def _load_refined_b1_b2(root: Path) -> dict[tuple[str, int], dict[str, float]]:
    rows = _read_csv(
        root / "runs" / "20260829" / "EXP-PHY-006_202608291923" / "gap_refinement_samples.csv"
    )
    selected: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (row["lane"], int(row["dataset_index"]))
        if key not in SAMPLES or int(row["lower_band_one_based"]) != 1:
            continue
        selected[key] = {
            "b1_max": float(row["refined_lower_edge_frequency"]),
            "b2_min": float(row["refined_upper_edge_frequency"]),
            "midpoint": float(row["refined_center"]),
            "signed_width": float(row["refined_signed_width"]),
        }
    missing = set(SAMPLES) - set(selected)
    if missing:
        raise ValueError(f"Missing refined B1-B2 records for {sorted(missing)}.")
    return selected


def _load_pair_counts(root: Path) -> dict[str, tuple[int, int]]:
    path = root / "runs" / "20260828" / "EXP-PHY-005_202608281751" / "gap_alignment_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: dict[str, tuple[int, int]] = {}
    for lane in ("binary", "graded"):
        metrics = payload["analytic"]["x"][lane]
        distribution = metrics["validation_pair_distribution"]
        selected[lane] = (int(distribution.get("1", 0)), int(metrics["validation_count"]))
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


def _plot_sample(
    structure_axis: plt.Axes,
    band_axis: plt.Axes,
    *,
    key: tuple[str, int],
    structure: np.ndarray,
    bands: np.ndarray,
    path_indices: np.ndarray,
    ticks: np.ndarray,
    tick_labels: list[str],
    prior: float,
    refined: dict[str, float],
    panel_label: str,
) -> None:
    structure_axis.imshow(
        structure,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        origin="lower",
    )
    structure_axis.set_title("Structure", fontsize=6.5, pad=2.0)
    structure_axis.set_xticks([])
    structure_axis.set_yticks([])
    for spine in structure_axis.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#666666")

    path_bands = bands[path_indices]
    x = np.arange(path_bands.shape[0])
    for band in range(2, path_bands.shape[1]):
        band_axis.plot(
            x,
            path_bands[:, band],
            color=COLORS["higher"],
            linewidth=0.75,
            alpha=0.55,
            label="B3-B10 real MPB" if band == 2 else None,
            zorder=2,
        )
    band_axis.plot(
        x, path_bands[:, 0], color=COLORS["b1"], linewidth=1.7, label="B1 real MPB", zorder=4
    )
    band_axis.plot(
        x, path_bands[:, 1], color=COLORS["b2"], linewidth=1.7, label="B2 real MPB", zorder=4
    )

    b1_max = float(refined["b1_max"])
    b2_min = float(refined["b2_min"])
    midpoint = float(refined["midpoint"])
    width = float(refined["signed_width"])
    band_axis.axhspan(
        min(b1_max, b2_min),
        max(b1_max, b2_min),
        facecolor=COLORS["overlap"],
        edgecolor=COLORS["overlap"],
        alpha=0.13,
        hatch="////" if width < 0 else None,
        linewidth=0.5,
        label="Refined B1-B2 overlap" if width < 0 else "Refined B1-B2 gap",
        zorder=1,
    )
    band_axis.axhline(
        midpoint,
        color=COLORS["midpoint"],
        linestyle="-.",
        linewidth=1.0,
        label="Refined B1-B2 midpoint",
        zorder=3,
    )
    band_axis.axhline(
        prior,
        color=COLORS["prior"],
        linestyle="--",
        linewidth=1.4,
        label=r"Prior $x=1/(2n_{\mathrm{eff}})$",
        zorder=5,
    )
    band_axis.axhline(b1_max, color=COLORS["b1"], linestyle=":", linewidth=0.7, alpha=0.8)
    band_axis.axhline(b2_min, color=COLORS["b2"], linestyle=":", linewidth=0.7, alpha=0.8)

    error = 100.0 * abs(prior - midpoint) / abs(midpoint)
    band_axis.text(
        0.985,
        0.045,
        f"prior = {prior:.4f}\nB1-B2 midpoint = {midpoint:.4f}\n"
        f"relative offset = {error:.2f}%\nsigned width = {width:.4f}",
        transform=band_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.0,
        color=COLORS["text"],
    )
    band_axis.text(
        0.02,
        0.06,
        "B1-B2 overlap\nnot a complete gap" if width < 0 else "B1-B2 complete gap",
        transform=band_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=COLORS["prior"] if width < 0 else COLORS["midpoint"],
        fontweight="bold",
    )
    band_axis.set_xticks(ticks)
    band_axis.set_xticklabels(tick_labels)
    band_axis.set_xlim(float(ticks[0]), float(ticks[-1]))
    band_axis.set_ylim(-0.02, float(np.max(path_bands)) + 0.025)
    band_axis.set_xlabel("k path")
    band_axis.set_ylabel(r"Normalized frequency $\omega a/(2\pi c)$", labelpad=3.0)
    band_axis.set_title(f"{panel_label}) {key[0].title()} validation | index {key[1]}", loc="left")
    band_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.65, zorder=0)
    for tick in ticks[1:-1]:
        band_axis.axvline(float(tick), color=COLORS["grid"], linewidth=0.7, zorder=0)
    band_axis.spines["right"].set_visible(False)
    band_axis.spines["top"].set_visible(False)
    band_axis.tick_params(width=0.7, length=3)


def _write_source_data(
    path: Path,
    *,
    bands: dict[tuple[str, int], np.ndarray],
    path_indices: np.ndarray,
    path_coordinates: np.ndarray,
    priors: dict[tuple[str, int], float],
    refined: dict[tuple[str, int], dict[str, float]],
    pair_counts: dict[str, tuple[int, int]],
) -> None:
    fields = [
        "lane",
        "dataset_index",
        "path_position",
        "kx",
        "ky",
        "band",
        "mpb_frequency",
        "formula_x_frequency",
        "refined_b1_max",
        "refined_b2_min",
        "refined_midpoint",
        "refined_signed_width",
        "validation_b1_b2_count",
        "validation_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in SAMPLES:
            detail = refined[key]
            selected_count, total_count = pair_counts[key[0]]
            selected_bands = bands[key][path_indices]
            for path_position, (kx, ky) in enumerate(path_coordinates):
                for band in range(selected_bands.shape[1]):
                    writer.writerow(
                        {
                            "lane": key[0],
                            "dataset_index": key[1],
                            "path_position": path_position,
                            "kx": f"{kx:.12g}",
                            "ky": f"{ky:.12g}",
                            "band": band + 1,
                            "mpb_frequency": f"{selected_bands[path_position, band]:.12g}",
                            "formula_x_frequency": f"{priors[key]:.12g}",
                            "refined_b1_max": f"{detail['b1_max']:.12g}",
                            "refined_b2_min": f"{detail['b2_min']:.12g}",
                            "refined_midpoint": f"{detail['midpoint']:.12g}",
                            "refined_signed_width": f"{detail['signed_width']:.12g}",
                            "validation_b1_b2_count": selected_count,
                            "validation_count": total_count,
                        }
                    )


def _write_structure_manifest(
    path: Path,
    *,
    root: Path,
    structures: dict[tuple[str, int], np.ndarray],
    structure_fields: dict[tuple[str, int], str],
) -> None:
    fields = ["lane", "dataset_index", "dataset", "grid_field", "shape", "minimum", "maximum"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in SAMPLES:
            grid = structures[key]
            writer.writerow(
                {
                    "lane": key[0],
                    "dataset_index": key[1],
                    "dataset": _dataset_path(root, key[0]).relative_to(root).as_posix(),
                    "grid_field": structure_fields[key],
                    "shape": "x".join(str(value) for value in grid.shape),
                    "minimum": f"{float(np.min(grid)):.12g}",
                    "maximum": f"{float(np.max(grid)):.12g}",
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

    bands, structures, structure_fields, k_points = _load_sample_data(root)
    priors = _load_formula_x(root)
    refined = _load_refined_b1_b2(root)
    pair_counts = _load_pair_counts(root)
    path_indices, ticks, tick_labels = _high_symmetry_path(k_points)
    path_coordinates = k_points[path_indices]

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "legend.fontsize": 6.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    figure = plt.figure(figsize=(7.20, 5.55), constrained_layout=False)
    outer = figure.add_gridspec(2, 2, wspace=0.30, hspace=0.34)
    figure.subplots_adjust(left=0.045, right=0.985, top=0.865, bottom=0.125)
    band_axes: list[plt.Axes] = []
    for position, key in enumerate(SAMPLES):
        inner = outer[position // 2, position % 2].subgridspec(
            1, 2, width_ratios=(0.22, 0.78), wspace=0.48
        )
        structure_axis = figure.add_subplot(inner[0, 0])
        band_axis = figure.add_subplot(inner[0, 1])
        band_axes.append(band_axis)
        _plot_sample(
            structure_axis,
            band_axis,
            key=key,
            structure=structures[key],
            bands=bands[key],
            path_indices=path_indices,
            ticks=ticks,
            tick_labels=tick_labels,
            prior=priors[key],
            refined=refined[key],
            panel_label="abcd"[position],
        )
    handles, labels = band_axes[0].get_legend_handles_labels()
    legend_items = dict(zip(labels, handles, strict=True))
    legend_order = [
        "B1 real MPB",
        "B2 real MPB",
        "B3-B10 real MPB",
        "Refined B1-B2 overlap",
        "Refined B1-B2 midpoint",
        r"Prior $x=1/(2n_{\mathrm{eff}})$",
    ]
    figure.legend(
        [legend_items[label] for label in legend_order],
        legend_order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=6,
        fontsize=5.5,
        columnspacing=1.0,
        handlelength=2.5,
    )
    binary_count, binary_total = pair_counts["binary"]
    graded_count, graded_total = pair_counts["graded"]
    figure.suptitle(
        "Why the analytic prior maps to B1-B2: low-order region, not a complete gap",
        fontsize=8.6,
        fontweight="bold",
        y=0.988,
    )
    figure.text(
        0.5,
        0.948,
        f"Validation pair selection: Binary {binary_count}/{binary_total}; "
        f"Graded {graded_count}/{graded_total}",
        ha="center",
        va="top",
        fontsize=6.5,
        color=COLORS["text"],
    )
    base = output_dir / "b1_b2_prior_real_mpb_band_diagram"
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    _strip_svg_trailing_whitespace(base.with_suffix(".svg"))
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    _write_source_data(
        output_dir / "b1_b2_prior_real_mpb_source_data.csv",
        bands=bands,
        path_indices=path_indices,
        path_coordinates=path_coordinates,
        priors=priors,
        refined=refined,
        pair_counts=pair_counts,
    )
    _write_structure_manifest(
        output_dir / "b1_b2_prior_structure_manifest.csv",
        root=root,
        structures=structures,
        structure_fields=structure_fields,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
