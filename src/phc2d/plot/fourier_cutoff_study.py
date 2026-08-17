"""Blue-white diagnostics for paired Fourier-cutoff studies.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_member(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        structure = json.loads(_decode(handle["structure_json"][0]))
        keys = [name for name in handle if name.endswith("_256") and name.startswith("material_")]
        if len(keys) != 1:
            raise ValueError(f"expected one 256-grid in {path}, got {keys}")
        return {
            "grid": np.asarray(handle[keys[0]][0], dtype=np.float64),
            "bands": np.asarray(handle["Wh"][0], dtype=np.float64),
            "stats": dict(structure.get("metadata", {}).get("stats", {})),
            "structure": structure,
        }


def _save(fig: Any, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def plot_fourier_cutoff_study(root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Load Binary/Graded K05..K20 members and write paired comparisons."""

    import matplotlib.pyplot as plt

    campaign = Path(root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cutoffs = np.arange(5, 21, dtype=int)
    data = {
        lane: [_load_member(campaign / f"{lane}_k{cutoff:02d}" / "dataset.h5") for cutoff in cutoffs]
        for lane in ("binary", "graded")
    }
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(4, 8, figsize=(15.5, 7.6), constrained_layout=True)
    for lane_index, lane in enumerate(("binary", "graded")):
        for offset, (cutoff, row) in enumerate(zip(cutoffs, data[lane])):
            axis = axes[lane_index * 2 + offset // 8, offset % 8]
            axis.imshow(row["grid"], cmap="Blues", vmin=0.0, vmax=1.0, origin="lower")
            axis.set_title(f"{lane[0].upper()} · K={cutoff}", fontsize=8)
            axis.set_xticks([]); axis.set_yticks([])
    _save(fig, out / "paired_structures_k05_k20")
    plt.close(fig)

    for lane in ("binary", "graded"):
        fig, axes = plt.subplots(2, 8, figsize=(15.5, 4.0), constrained_layout=True)
        for offset, (cutoff, row) in enumerate(zip(cutoffs, data[lane])):
            axis = axes[offset // 8, offset % 8]
            axis.imshow(row["grid"], cmap="Blues", vmin=0.0, vmax=1.0, origin="lower")
            axis.set_title(f"K={cutoff}", fontsize=9)
            axis.set_xticks([]); axis.set_yticks([])
        fig.suptitle(f"{lane.title()} structures · Fourier cutoff K=5–20", fontsize=13)
        _save(fig, out / f"{lane}_structures_k05_k20")
        plt.close(fig)

    rows: list[dict[str, Any]] = []
    for lane in ("binary", "graded"):
        reference = data[lane][-1]["bands"]
        for cutoff, member in zip(cutoffs, data[lane]):
            bands = member["bands"]
            relative = np.abs(bands - reference) / np.maximum(np.abs(reference), 1.0e-8)
            rows.append({
                "lane": lane, "fourier_cutoff": int(cutoff),
                "mode_count": len(member["structure"]["parameters"]["modes"]),
                "material_fraction": float(member["grid"].mean()),
                "gray_fraction": float(np.mean((member["grid"] > 0.01) & (member["grid"] < 0.99))),
                "boundary_variation_per_a": float(member["stats"].get("boundary_variation_per_a", np.nan)),
                "spectral_second_moment": float(member["stats"].get("spectral_second_moment", np.nan)),
                "wh_mre_vs_k20": float(relative.mean()),
                "wh_max_abs_vs_k20": float(np.max(np.abs(bands - reference))),
                "band_mre_vs_k20": relative.mean(axis=0).tolist(),
            })

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    colors = {"binary": "#1D4ED8", "graded": "#60A5FA"}
    for lane in ("binary", "graded"):
        lane_rows = [row for row in rows if row["lane"] == lane]
        for axis, key, label in (
            (axes[0, 0], "material_fraction", "Material fraction"),
            (axes[0, 1], "boundary_variation_per_a", "Boundary variation / a"),
            (axes[1, 0], "spectral_second_moment", "Spectral second moment"),
            (axes[1, 1], "wh_mre_vs_k20", "Wh MRE vs K=20"),
        ):
            axis.plot(cutoffs, [row[key] for row in lane_rows], marker="o", ms=3.5, lw=1.5, color=colors[lane], label=lane.title())
            axis.set(xlabel="Fourier cutoff K", ylabel=label, xticks=[5, 8, 12, 16, 20])
    axes[0, 0].legend(frameon=False)
    _save(fig, out / "structure_and_band_effects")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for axis, lane in zip(axes, ("binary", "graded")):
        matrix = np.asarray([row["band_mre_vs_k20"] for row in rows if row["lane"] == lane])
        image = axis.imshow(matrix, cmap="Blues", aspect="auto", origin="lower")
        axis.set(title=lane.title(), xlabel="Band", ylabel="Fourier cutoff K")
        axis.set_xticks(np.arange(10), np.arange(1, 11))
        axis.set_yticks(np.arange(len(cutoffs)), cutoffs)
        fig.colorbar(image, ax=axis, label="MRE vs K=20")
    _save(fig, out / "bandwise_mre_heatmap")
    plt.close(fig)

    with (out / "cutoff_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in rows[0] if key != "band_mre_vs_k20"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in rows)
    summary = {"status": "complete", "cutoffs": cutoffs.tolist(), "member_count": len(rows), "rows": rows}
    (out / "cutoff_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path); parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = plot_fourier_cutoff_study(args.root, args.output_dir)
    print(json.dumps({"status": result["status"], "member_count": result["member_count"]}))


if __name__ == "__main__":
    main()
