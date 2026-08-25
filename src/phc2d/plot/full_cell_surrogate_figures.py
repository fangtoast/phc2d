"""Publication figures for Full-Wh spectral and response-sensitivity evidence.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PALETTE = {
    "truth": "#272727",
    "convnext": "#7884B4",
    "joint": "#F0C0CC",
    "convnext_dark": "#484878",
    "joint_dark": "#9A4D8E",
    "neutral": "#A8A8A8",
}


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.3,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    return plt


def _save(fig: Any, stem: Path, *, formats: Sequence[str], dpi: int) -> list[Path]:
    paths: list[Path] = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        kwargs: dict[str, Any] = {"facecolor": "white"}
        if suffix in {"png", "tif", "tiff"}:
            kwargs["dpi"] = int(dpi)
        if suffix in {"tif", "tiff"}:
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def _plot_bands(
    ax: Any,
    values: np.ndarray,
    path_indices: np.ndarray,
    *,
    color: str,
    linestyle: str = "-",
    label: str | None = None,
    alpha: float = 0.92,
) -> None:
    selected = np.asarray(values, dtype=np.float64)[path_indices]
    x = np.arange(len(path_indices))
    for band in range(selected.shape[1]):
        ax.plot(
            x,
            selected[:, band],
            color=color,
            linestyle=linestyle,
            linewidth=0.72,
            alpha=alpha,
            label=label if band == 0 else None,
        )


def render_spectral_summary(
    lane_payloads: Mapping[str, Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Render full-test ECDF and per-band evidence for both K5 lanes."""

    plt = _matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 112 / 25.4), squeeze=False)
    for row_index, lane in enumerate(("B", "G")):
        payload = lane_payloads[lane]
        ax = axes[row_index, 0]
        for model, color, label in (
            ("convnext", PALETTE["convnext_dark"], "ConvNeXt"),
            ("joint", PALETTE["joint_dark"], "Joint"),
        ):
            values = np.sort(np.asarray(payload[f"{model}_sample_mre_pct"], dtype=np.float64))
            ax.plot(values, (np.arange(values.size) + 1) / values.size, color=color, linewidth=1.4, label=label)
        for selected in (item for item in selected_rows if item["lane"] == lane):
            ax.axvline(float(selected["difficulty_mre_pct"]), color=PALETTE["neutral"], linewidth=0.6, alpha=0.7)
            ax.text(
                float(selected["difficulty_mre_pct"]),
                0.04,
                f"{selected['quantile_role']} {selected['quantile_label']}",
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=5.0,
                color="#606060",
            )
        ax.set_xlabel("Bands 2–10 sample MRE vs ground truth (%)")
        ax.set_ylabel("Empirical cumulative fraction")
        ax.set_title(f"{payload['display_name']} · n={len(values):,}")
        ax.grid(alpha=0.16)
        ax.legend(loc="lower right")

        ax = axes[row_index, 1]
        bands = np.arange(1, 11)
        ax.plot(
            bands,
            np.asarray(payload["convnext_per_band_mre_pct"]),
            marker="o",
            markersize=3.0,
            color=PALETTE["convnext_dark"],
            linewidth=1.2,
            label="ConvNeXt",
        )
        ax.plot(
            bands,
            np.asarray(payload["joint_per_band_mre_pct"]),
            marker="s",
            markersize=2.8,
            color=PALETTE["joint_dark"],
            linewidth=1.2,
            label="Joint",
        )
        ax.axvspan(0.5, 1.5, color="#E4E4F0", alpha=0.8, zorder=-5)
        ax.text(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0, "context only", ha="center", va="top", fontsize=4.8)
        ax.set_xticks(bands)
        ax.set_xlabel("Band (one-based)")
        ax.set_ylabel("Full-BZ MRE vs ground truth (%)")
        ax.set_title(f"{payload['display_name']} · per-band error vs ground truth")
        ax.grid(alpha=0.16)
        ax.legend()
    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.suptitle("EXP-PHY-004 · Errors against frozen real-MPB ground truth", fontsize=8, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Ground truth: frozen HDF5 real-MPB Wh[idx_test]. Gray markers: quantiles of mean(ConvNeXt, Joint) sample MRE.",
        ha="center",
        fontsize=5.1,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.15, top=0.90, wspace=0.28, hspace=0.42)
    paths = _save(fig, figures_dir / "spectral_full_test_summary", formats=formats, dpi=dpi)
    plt.close(fig)
    return paths


def render_band_plates(
    lane_payloads: Mapping[str, Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Render separate raw spectral plates followed by model-minus-MPB differences."""

    plt = _matplotlib()
    outputs: list[Path] = []
    for lane in ("B", "G"):
        payload = lane_payloads[lane]
        rows = [item for item in selected_rows if item["lane"] == lane]
        path_indices = np.asarray(payload["path_indices"], dtype=np.int64)
        ticks = tuple(int(value) for value in payload["path_ticks"])
        tick_labels = tuple(str(value) for value in payload["path_tick_labels"])
        all_values = np.concatenate(
            [
                np.asarray(payload[source])[int(row["test_position"])][path_indices].reshape(-1)
                for row in rows
                for source in ("target", "convnext", "joint")
            ]
        )
        margin = 0.04 * max(float(np.ptp(all_values)), 1.0e-3)
        limits = (float(np.min(all_values) - margin), float(np.max(all_values) + margin))

        fig, axes = plt.subplots(len(rows), 5, figsize=(183 / 25.4, 126 / 25.4), squeeze=False)
        for row_index, selected in enumerate(rows):
            pos = int(selected["test_position"])
            ax = axes[row_index, 0]
            image = ax.imshow(payload["epsilon_by_test_position"][pos], origin="lower", cmap="Blues", vmin=1.0, vmax=8.9)
            ax.set_title(
                f"{selected['sample_label']} · {selected['quantile_role']} {selected['quantile_label']}\n"
                r"$\varepsilon_r$ map"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for col, source, title, color in (
                (1, "target", "Ground truth\n(stored real-MPB)", PALETTE["truth"]),
                (2, "convnext", "ConvNeXt", PALETTE["convnext_dark"]),
                (3, "joint", "Joint", PALETTE["joint_dark"]),
            ):
                ax = axes[row_index, col]
                _plot_bands(ax, payload[source][pos], path_indices, color=color)
                ax.set_title(title)
                ax.set_ylim(*limits)
                ax.set_xticks(ticks, tick_labels)
                ax.grid(axis="x", alpha=0.18)
                if col == 1:
                    ax.set_ylabel(r"Frequency $a/\lambda$")
            ax = axes[row_index, 4]
            _plot_bands(ax, payload["target"][pos], path_indices, color=PALETTE["truth"], label="Ground truth")
            _plot_bands(
                ax,
                payload["convnext"][pos],
                path_indices,
                color=PALETTE["convnext_dark"],
                linestyle="--",
                label="ConvNeXt",
            )
            _plot_bands(
                ax,
                payload["joint"][pos],
                path_indices,
                color=PALETTE["joint_dark"],
                linestyle=":",
                label="Joint",
            )
            ax.set_title("Ground truth + predictions")
            ax.set_ylim(*limits)
            ax.set_xticks(ticks, tick_labels)
            ax.grid(axis="x", alpha=0.18)
            if row_index == 0:
                ax.legend(fontsize=4.8, loc="upper left")
        for label, ax in zip("abcde", axes[0], strict=True):
            ax.text(-0.12, 1.12, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        fig.suptitle(
            f"{payload['display_name']} · ground-truth bands shown before predictions and differences",
            fontsize=8,
            fontweight="bold",
        )
        fig.text(0.12, 0.018, r"Dielectric maps use a fixed $\varepsilon_r=1.0$–$8.9$ scale.", ha="center", fontsize=5.0)
        fig.subplots_adjust(left=0.045, right=0.99, bottom=0.07, top=0.90, wspace=0.26, hspace=0.38)
        outputs += _save(fig, figures_dir / f"spectral_raw_{lane.lower()}", formats=formats, dpi=dpi)
        plt.close(fig)

        signed_errors: list[np.ndarray] = []
        for selected in rows:
            pos = int(selected["test_position"])
            truth = np.asarray(payload["target"][pos], dtype=np.float64)
            for source in ("convnext", "joint"):
                signed_errors.append(100.0 * (np.asarray(payload[source][pos]) - truth) / np.maximum(np.abs(truth), 1.0e-8))
        bound = float(np.percentile(np.abs(np.concatenate([item.reshape(-1) for item in signed_errors])), 99.0))
        bound = max(bound, 0.1)
        fig, axes = plt.subplots(len(rows), 3, figsize=(183 / 25.4, 126 / 25.4), squeeze=False)
        images: list[Any] = []
        for row_index, selected in enumerate(rows):
            pos = int(selected["test_position"])
            truth = np.asarray(payload["target"][pos], dtype=np.float64)
            ax = axes[row_index, 0]
            _plot_bands(ax, truth, path_indices, color=PALETTE["truth"], label="Ground truth")
            _plot_bands(ax, payload["convnext"][pos], path_indices, color=PALETTE["convnext_dark"], linestyle="--", label="ConvNeXt")
            _plot_bands(ax, payload["joint"][pos], path_indices, color=PALETTE["joint_dark"], linestyle=":", label="Joint")
            ax.set_ylim(*limits)
            ax.set_xticks(ticks, tick_labels)
            ax.set_ylabel(r"Frequency $a/\lambda$")
            ax.set_title(
                f"{selected['sample_label']} · overlay\n"
                f"C={float(selected['convnext_mre_pct']):.3f}% · J={float(selected['joint_mre_pct']):.3f}%"
            )
            ax.grid(axis="x", alpha=0.18)
            if row_index == 0:
                ax.legend(fontsize=4.8)
            for col, source, title in (
                (1, "convnext", "ConvNeXt − ground truth"),
                (2, "joint", "Joint − ground truth"),
            ):
                signed = 100.0 * (np.asarray(payload[source][pos], dtype=np.float64) - truth) / np.maximum(np.abs(truth), 1.0e-8)
                image = axes[row_index, col].imshow(
                    signed.T,
                    origin="lower",
                    aspect="auto",
                    cmap="RdBu_r",
                    vmin=-bound,
                    vmax=bound,
                    interpolation="nearest",
                )
                images.append(image)
                axes[row_index, col].set_title(title)
                axes[row_index, col].set_xlabel("Flattened full-BZ k index")
                axes[row_index, col].set_ylabel("Band")
                axes[row_index, col].set_yticks(np.arange(10), np.arange(1, 11))
        colorbar_ax = fig.add_axes([0.925, 0.15, 0.012, 0.68])
        cbar = fig.colorbar(images[-1], cax=colorbar_ax)
        cbar.set_label("Signed relative frequency error (%)")
        for label, ax in zip("abc", axes[0], strict=True):
            ax.text(-0.10, 1.12, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        fig.suptitle(
            f"{payload['display_name']} · full-BZ differences from real-MPB ground truth",
            fontsize=8,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.07, right=0.895, bottom=0.07, top=0.89, wspace=0.34, hspace=0.58)
        outputs += _save(fig, figures_dir / f"spectral_difference_{lane.lower()}", formats=formats, dpi=dpi)
        plt.close(fig)
    return outputs


def render_eigenfield_plates(
    field_records: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Render truth-only real-MPB eigenfield plates for the selected samples."""

    plt = _matplotlib()
    outputs: list[Path] = []
    for lane in ("B", "G"):
        rows = [item for item in field_records if item["lane"] == lane]
        fig, axes = plt.subplots(1, len(rows), figsize=(183 / 25.4, 68 / 25.4), squeeze=False)
        images = []
        for ax, row in zip(axes[0], rows, strict=True):
            image = ax.imshow(row["energy"], origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
            images.append(image)
            ax.contour(row["epsilon"], levels=[0.5 * (1.0 + 8.9)], colors="#22D7E6", linewidths=0.35)
            ax.set_title(
                f"{row['sample_label']} · b{row['band_index_one_based']} · k={row['k_index']}\n"
                f"replay MRE={100.0 * float(row['frequency_mre']):.2e}%"
            )
            ax.set_xticks([])
            ax.set_yticks([])
        cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), fraction=0.018, pad=0.018)
        cbar.set_label(r"Per-sample normalized $\varepsilon|E|^2$")
        fig.suptitle(
            f"K5-{lane} · solver-reference real-MPB upper-gap-edge TE eigenfields",
            fontsize=8,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.015,
            "Reference fields come from real-MPB replay; the Full-Wh surrogate predicts bands, not fields.",
            ha="center",
            fontsize=5.4,
        )
        fig.subplots_adjust(left=0.025, right=0.91, bottom=0.09, top=0.82, wspace=0.10)
        outputs += _save(fig, figures_dir / f"real_mpb_eigenfield_{lane.lower()}", formats=formats, dpi=dpi)
        plt.close(fig)
    return outputs


def _normalized_response_groups(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        grouped.setdefault((str(row["lane"]), str(row["sample_label"])), {})[str(row["source"])] = row
    for key, values in grouped.items():
        if set(values) != {"real_mpb", "convnext", "joint"}:
            raise ValueError(f"response group {key} lacks real_mpb/convnext/joint cases")
        common_max = max(float(np.max(item["electric_energy"])) for item in values.values())
        if not np.isfinite(common_max) or common_max <= 0.0:
            raise ValueError(f"response group {key} has no positive finite near-field energy")
        for item in values.values():
            item["near_common_normalized"] = np.asarray(item["electric_energy"], dtype=np.float64) / common_max
            far = np.asarray(item["far_radial_flux"], dtype=np.float64)
            integral = float(np.trapezoid(far, x=np.asarray(item["far_angles_rad"], dtype=np.float64)))
            if not np.isfinite(integral) or integral <= 0.0:
                raise ValueError(f"response group {key} has no positive finite far-field integral")
            item["far_unit_area"] = far / integral
    return grouped


def render_response_plates(
    response_records: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Render raw Meep responses first, then response-difference plates."""

    plt = _matplotlib()
    groups = _normalized_response_groups(response_records)
    outputs: list[Path] = []
    for lane in ("B", "G"):
        lane_groups = [(key, value) for key, value in groups.items() if key[0] == lane]
        lane_groups.sort(key=lambda item: item[0][1])
        fig = plt.figure(figsize=(183 / 25.4, 142 / 25.4))
        grid = fig.add_gridspec(len(lane_groups), 6, wspace=0.40, hspace=0.42)
        for row_index, ((_, sample_label), values) in enumerate(lane_groups):
            for pair_index, (source, label) in enumerate(
                (
                    ("real_mpb", "Reference: real-MPB f"),
                    ("convnext", "ConvNeXt-selected f"),
                    ("joint", "Joint-selected f"),
                )
            ):
                record = values[source]
                near_ax = fig.add_subplot(grid[row_index, 2 * pair_index])
                extent = (record["x"][0], record["x"][-1], record["y"][0], record["y"][-1])
                near_ax.imshow(record["near_common_normalized"], origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
                near_ax.contour(record["x"], record["y"], record["epsilon"], levels=[0.5 * (1.0 + 8.9)], colors="#22D7E6", linewidths=0.25)
                near_ax.set_xticks([])
                near_ax.set_yticks([])
                near_ax.set_title(f"{sample_label} · {label}\nnear · f={float(record['frequency']):.4f}", fontsize=5.2)
                far_ax = fig.add_subplot(grid[row_index, 2 * pair_index + 1], projection="polar")
                angles = np.r_[record["far_angles_rad"], record["far_angles_rad"][0] + 2.0 * np.pi]
                radial = np.r_[record["far_unit_area"], record["far_unit_area"][0]]
                far_ax.plot(angles, radial, color=PALETTE[source if source != "real_mpb" else "truth"], linewidth=0.8)
                far_ax.set_theta_zero_location("E")
                far_ax.set_theta_direction(1)
                far_ax.set_rticks([])
                far_ax.set_thetagrids([0, 90, 180, 270], labels=["+x", "+y", "−x", "−y"])
                far_ax.tick_params(labelsize=4.2, pad=0)
                far_ax.set_title("far shape", fontsize=5.2, pad=6)
        fig.suptitle(
            f"K5-{lane} · real-Meep responses at reference and surrogate-selected frequencies",
            fontsize=8,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.01,
            "All columns are real-Meep solves. Surrogates predict bands, select (band, k, f), and supply only f to Meep.",
            ha="center",
            fontsize=5.4,
        )
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.91)
        outputs += _save(fig, figures_dir / f"meep_response_raw_{lane.lower()}", formats=formats, dpi=dpi)
        plt.close(fig)

        fig = plt.figure(figsize=(183 / 25.4, 126 / 25.4))
        grid = fig.add_gridspec(len(lane_groups), 4, wspace=0.38, hspace=0.42)
        for row_index, ((_, sample_label), values) in enumerate(lane_groups):
            truth = values["real_mpb"]
            diffs = [values[source]["near_common_normalized"] - truth["near_common_normalized"] for source in ("convnext", "joint")]
            bound = max(float(np.max(np.abs(item))) for item in diffs)
            bound = max(bound, 1.0e-6)
            for col, (source, title) in enumerate(
                (
                    ("convnext", "ConvNeXt-selected f − reference f"),
                    ("joint", "Joint-selected f − reference f"),
                )
            ):
                ax = fig.add_subplot(grid[row_index, col])
                image = ax.imshow(diffs[col], origin="lower", cmap="RdBu_r", vmin=-bound, vmax=bound)
                ax.set_title(f"{sample_label} · near Δ\n{title}", fontsize=5.2)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 1:
                    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
                    cbar.set_label("Common-scale energy Δ", fontsize=4.8)
            far_ax = fig.add_subplot(grid[row_index, 2], projection="polar")
            for source, color, linestyle, label in (
                ("real_mpb", PALETTE["truth"], "-", "Reference f"),
                ("convnext", PALETTE["convnext_dark"], "--", "ConvNeXt-selected f"),
                ("joint", PALETTE["joint_dark"], ":", "Joint-selected f"),
            ):
                record = values[source]
                angles = np.r_[record["far_angles_rad"], record["far_angles_rad"][0] + 2.0 * np.pi]
                radial = np.r_[record["far_unit_area"], record["far_unit_area"][0]]
                far_ax.plot(angles, radial, color=color, linestyle=linestyle, linewidth=0.8, label=label)
            far_ax.set_theta_zero_location("E")
            far_ax.set_theta_direction(1)
            far_ax.set_rticks([])
            far_ax.set_thetagrids([0, 90, 180, 270], labels=["+x", "+y", "−x", "−y"])
            far_ax.tick_params(labelsize=4.2, pad=0)
            if row_index == 0:
                far_ax.legend(fontsize=4.2, bbox_to_anchor=(1.15, 1.20))
            metric_ax = fig.add_subplot(grid[row_index, 3])
            metric_ax.axis("off")
            lines = [f"{sample_label} response differences"]
            for source, label in (("convnext", "ConvNeXt"), ("joint", "Joint")):
                metrics = values[source]["difference_metrics"]
                lines.extend(
                    [
                        f"{label}:",
                        f"  |Δf|/f = {100.0 * float(metrics['frequency_relative_error']):.3f}%",
                        f"  near rel-L2 = {float(metrics['near_relative_l2']):.3f}",
                        f"  near cosine = {float(metrics['near_cosine_similarity']):.3f}",
                        f"  far angular L1 = {float(metrics['far_angular_l1']):.3f}",
                        f"  lobe shift = {float(metrics['far_lobe_shift_deg']):.1f}°",
                    ]
                )
            metric_ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=5.2, family="monospace")
        fig.suptitle(
            f"K5-{lane} · real-Meep response differences caused only by selected-frequency changes",
            fontsize=8,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.02, right=0.985, bottom=0.04, top=0.91)
        outputs += _save(fig, figures_dir / f"meep_response_difference_{lane.lower()}", formats=formats, dpi=dpi)
        plt.close(fig)
    return outputs


__all__ = [
    "render_band_plates",
    "render_eigenfield_plates",
    "render_response_plates",
    "render_spectral_summary",
]
