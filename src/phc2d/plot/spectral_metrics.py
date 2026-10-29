"""Plot matched native metric ranks, band errors and seed-specific ECDFs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np


def plot_spectral_metrics(run: Path, output: Path) -> list[Path]:
    """Plot available complete native groups; never pool seed quantiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    def read(path):
        with path.open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    summary = [r for r in read(run / "three_seed_summary.csv")
               if r["comparison_role"] == "native" and r["band_group"] == "all_bands" and r["status"] == "complete"]
    if not summary:
        raise ValueError("no complete native groups to plot")
    domains = sorted({r["target_domain"] for r in summary})
    models = sorted({r["model"] for r in summary})
    complete_pairs = {(r["model"], r["target_domain"]) for r in summary}
    aliases = {"global_local_band_operator_v1": "Historical GL 3x3",
               "tfno2d_band_v3": "Historical TFNO", "convnext_band_v1": "Historical ConvNeXt"}
    palette = ["#0F4D92", "#B64342", "#42949E", "#9A4D8E", "#767676"]
    palette.extend(plt.get_cmap("tab20").colors)
    colors = {model: palette[i % len(palette)] for i, model in enumerate(models)}
    output.mkdir(parents=True, exist_ok=True)
    exported = []

    def save(fig, name):
        for suffix in ("svg", "png"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            exported.append(path)
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.4, "legend.frameon": False, "svg.fonttype": "none"}):
        metrics = ["mre_pct", "relative_l2_mean_pct", "rmse_global", "relative_l2_p95_pct"]
        labels = ["MRE", "Rel. L2", "RMSE", "P95 L2"]
        fig, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), max(3.6, .5 * len(models))), squeeze=False, layout="constrained")
        for ax, domain in zip(axes[0], domains):
            rows = [r for r in summary if r["target_domain"] == domain]
            data = np.array([[float(r[m + "_mean"]) for m in metrics] for r in rows])
            ranks = np.column_stack([rankdata(data[:, j]) for j in range(len(metrics))])
            ax.imshow(ranks, cmap="Blues_r", vmin=1, vmax=max(2, len(rows)), aspect="auto")
            ax.set_xticks(range(len(metrics)), labels)
            ax.set_yticks(range(len(rows)), [aliases.get(r["model"], r["model"]) for r in rows])
            for i in range(len(rows)):
                for j in range(len(metrics)):
                    ax.text(j, i, f"{ranks[i,j]:g}", ha="center", va="center", color="white" if ranks[i,j] == 1 else "#202020")
            ax.set_title(domain.replace("_", " ") + " / native")
        fig.suptitle("Available complete groups: metric ranks (1 = lowest error)")
        save(fig, "native_metric_ranks")

        band_rows = [r for r in read(run / "per_band_metrics.csv") if r["comparison_role"] == "native" and (r["model"], r["target_domain"]) in complete_pairs]
        fig, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), 3.8), squeeze=False, layout="constrained")
        for ax, domain in zip(axes[0], domains):
            for model in models:
                rows = [r for r in band_rows if r["target_domain"] == domain and r["model"] == model]
                bands = sorted({int(r["band_one_based"]) for r in rows})
                if not bands:
                    continue
                data = np.array([[float(r["relative_l2_mean_pct"]) for r in rows if int(r["band_one_based"]) == b] for b in bands])
                mean, sd = data.mean(axis=1), data.std(axis=1, ddof=1)
                ax.plot(bands, mean, color=colors[model], marker="o", ms=3, label=aliases.get(model, model))
                ax.fill_between(bands, mean - sd, mean + sd, color=colors[model], alpha=.14)
            ax.set(xlabel="Band index", ylabel="Relative L2 error (%)", title=domain.replace("_", " "))
            ax.legend(fontsize=8)
        fig.suptitle("Native per-band error: three-seed mean and sample SD")
        save(fig, "native_per_band_l2")

        audit = [r for r in read(run / "cell_audit.csv") if r["status"] == "computed" and r["comparison_role"] == "native" and (r["model"], r["target_domain"]) in complete_pairs]
        fig, axes = plt.subplots(1, len(domains), figsize=(5.2 * len(domains), 3.8), squeeze=False, layout="constrained")
        for ax, domain in zip(axes[0], domains):
            for row in audit:
                if row["target_domain"] != domain:
                    continue
                with gzip.open(run / "cells" / row["cell_id"] / "per_sample_metrics.csv.gz", "rt", encoding="utf-8") as stream:
                    values = sorted(float(r["relative_l2_pct"]) for r in csv.DictReader(stream) if r["band_group"] == "all_bands")
                label = aliases.get(row["model"], row["model"]) if int(row["seed"]) == 0 else None
                ax.plot(values, np.arange(1, len(values)+1)/len(values), color=colors[row["model"]], alpha=.65, lw=1.2, label=label)
            ax.set(xlabel="Sample relative L2 error (%)", ylabel="Empirical cumulative probability", title=domain.replace("_", " "), ylim=(0, 1))
            ax.legend(fontsize=8)
        fig.suptitle("Native sample error: each line is one seed (no seed pooling)")
        save(fig, "native_l2_ecdf")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for path in plot_spectral_metrics(args.run, args.output):
        print(path)


if __name__ == "__main__":
    main()
