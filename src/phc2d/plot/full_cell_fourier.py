"""Reusable diagnostic plots for unified-Fourier full-cell datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _largest_complete_gap(bands: np.ndarray) -> np.ndarray:
    lower = np.max(bands[:, :, :-1], axis=1)
    upper = np.min(bands[:, :, 1:], axis=1)
    return np.maximum(upper - lower, 0.0).max(axis=1)


def plot_full_cell_fourier_dataset(dataset: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write a sample atlas and distribution/band summary for one v2 HDF5 file."""

    import h5py
    import matplotlib.pyplot as plt

    source = Path(dataset)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as handle:
        bands = np.asarray(handle["Wh"], dtype=np.float64)
        sample_ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["sample_id"][:]]
        grid_keys = sorted(
            name for name in handle
            if name.startswith(("material_binary_grid_", "material_weight_grid_"))
        )
        if not grid_keys:
            raise ValueError("dataset has no full-resolution material grid")
        grid_key = max(grid_keys, key=lambda name: int(name.rsplit("_", 1)[1]))
        grids = np.asarray(handle[grid_key], dtype=np.float64)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False})
    chosen = np.linspace(0, len(grids) - 1, min(16, len(grids)), dtype=int)
    fig, axes = plt.subplots(4, 4, figsize=(8.2, 8.2), constrained_layout=True)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, chosen):
        axis.imshow(grids[index], cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
        axis.set_title(sample_ids[index], fontsize=7)
    atlas = out / "sample_atlas.png"
    fig.savefig(atlas, dpi=300, bbox_inches="tight")
    fig.savefig(out / "sample_atlas.svg", bbox_inches="tight")
    plt.close(fig)

    means = grids.mean(axis=(1, 2))
    gray = np.mean((grids > 0.01) & (grids < 0.99), axis=(1, 2))
    gaps = _largest_complete_gap(bands)
    x = np.arange(bands.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)
    axes[0, 0].hist(means, bins=12, color="#3B82F6", edgecolor="white")
    axes[0, 0].set(xlabel="Material fraction", ylabel="Samples")
    axes[0, 1].hist(gray, bins=12, color="#10B981", edgecolor="white")
    axes[0, 1].set(xlabel="Gray-pixel fraction", ylabel="Samples")
    median = np.median(bands, axis=0)
    lo, hi = np.quantile(bands, [0.1, 0.9], axis=0)
    for band in range(bands.shape[2]):
        axes[1, 0].fill_between(x, lo[:, band], hi[:, band], color="#93C5FD", alpha=0.18)
        axes[1, 0].plot(x, median[:, band], color="#2563EB", lw=0.8)
    axes[1, 0].set(xlabel="Full-BZ k-point index", ylabel="Frequency (a/λ)")
    axes[1, 1].hist(gaps, bins=12, color="#EF4444", edgecolor="white")
    axes[1, 1].set(xlabel="Largest complete adjacent-band gap", ylabel="Samples")
    summary_plot = out / "distribution_and_bands.png"
    fig.savefig(summary_plot, dpi=300, bbox_inches="tight")
    fig.savefig(out / "distribution_and_bands.svg", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset": str(source), "sample_count": int(len(grids)), "grid_key": grid_key,
        "band_shape": list(bands.shape), "finite": bool(np.all(np.isfinite(bands))),
        "ascending": bool(np.all(np.diff(bands, axis=-1) >= -1.0e-7)),
        "material_fraction_min": float(means.min()), "material_fraction_max": float(means.max()),
        "gray_fraction_min": float(gray.min()), "gray_fraction_max": float(gray.max()),
        "largest_complete_gap_max": float(gaps.max()),
    }
    (out / "plot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(plot_full_cell_fourier_dataset(args.dataset, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
