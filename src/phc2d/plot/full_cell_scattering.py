"""Meep near/far-field response galleries for selected full-cell structures.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.full_cell import FullCellStructureSpec
from phc2d.plot.full_cell_gallery import GapEdgeSelection, select_largest_gap_edge
from phc2d.runs import (
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_json,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.solvers.meep_scattering import MeepScatteringConfig, MeepScatteringResult


ROOT = Path(__file__).resolve().parents[3]
ScatteringSolverFn = Callable[..., MeepScatteringResult]


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def normalize_response_for_display(
    electric_energy: np.ndarray,
    far_radial_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize near and far response magnitudes independently for visualization."""

    near = np.asarray(electric_energy, dtype=np.float64)
    far = np.asarray(far_radial_flux, dtype=np.float64)
    if np.any(~np.isfinite(near)) or np.any(~np.isfinite(far)) or np.any(near < 0.0) or np.any(far < 0.0):
        raise ValueError("near/far response arrays must be finite and non-negative")
    near_max, far_max = float(np.max(near)), float(np.max(far))
    if near_max <= 0.0 or far_max <= 0.0:
        raise ValueError("near/far response arrays must contain positive values")
    return (near / near_max).astype(np.float32), (far / far_max).astype(np.float32)


def poynting_quiver_samples(
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    *,
    target_count: int,
    relative_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample and globally scale time-averaged Poynting vectors for arrows."""

    x_values, y_values = np.asarray(x), np.asarray(y)
    sx_values, sy_values = np.asarray(sx), np.asarray(sy)
    if sx_values.shape != (y_values.size, x_values.size) or sy_values.shape != sx_values.shape:
        raise ValueError("Poynting arrays must have shape [len(y), len(x)]")
    stride_x = max(1, int(np.ceil(x_values.size / int(target_count))))
    stride_y = max(1, int(np.ceil(y_values.size / int(target_count))))
    xs, ys = x_values[::stride_x], y_values[::stride_y]
    u, v = sx_values[::stride_y, ::stride_x], sy_values[::stride_y, ::stride_x]
    magnitude = np.hypot(u, v)
    maximum = float(np.max(magnitude))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Poynting field has no finite positive magnitude")
    mask = magnitude >= float(relative_threshold) * maximum
    u = np.where(mask, u / maximum, np.nan)
    v = np.where(mask, v / maximum, np.nan)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y, u, v


def _configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.titlesize": 6.3,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return plt


def _grid(count: int) -> tuple[int, int]:
    cols = min(4, max(1, int(np.ceil(np.sqrt(count)))))
    return int(np.ceil(count / cols)), cols


def _save(fig: Any, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    outputs: list[Path] = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", **({"dpi": int(dpi)} if suffix == "png" else {}))
        outputs.append(path)
    return outputs


def _plot_near_field(
    records: Sequence[Mapping[str, Any]],
    *,
    lane: str,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
    arrow_count: int,
    arrow_threshold: float,
) -> list[Path]:
    plt = _configure_matplotlib()
    rows, cols = _grid(len(records))
    fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, 183 / 25.4), squeeze=False)
    image = None
    for position, ax in enumerate(axes.flat):
        if position >= len(records):
            ax.axis("off")
            continue
        row = records[position]
        result = row["result"]
        extent = (result.x[0], result.x[-1], result.y[0], result.y[-1])
        image = ax.imshow(row["near_normalized"], origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
        ax.contour(result.x, result.y, result.epsilon, levels=[0.5 * (1.0 + 8.9)], colors="#7FDBFF", linewidths=0.35, alpha=0.75)
        qx, qy, qu, qv = poynting_quiver_samples(
            result.x,
            result.y,
            result.poynting_x,
            result.poynting_y,
            target_count=arrow_count,
            relative_threshold=arrow_threshold,
        )
        ax.quiver(qx, qy, qu, qv, color="white", pivot="mid", angles="xy", scale_units="xy", scale=1.9, width=0.004)
        ax.annotate(
            r"$k_{inc}$",
            xy=(0.20, 0.91),
            xytext=(0.04, 0.91),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "#00d4ff", "lw": 1.0},
            color="#00d4ff",
            fontsize=5.2,
            va="center",
        )
        ax.set_title(f"{row['sample_label']} · f={result.frequency:.3f} · {row['gap'].status.replace('_', ' ')}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        cax = fig.add_axes([0.925, 0.25, 0.016, 0.50])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.set_label(r"Normalized electric energy $\varepsilon|E|^2$")
    fig.suptitle(
        f"{lane.replace('_', ' ')} · Meep driven near field with time-averaged Poynting arrows",
        fontsize=8,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.03, top=0.93, wspace=0.10, hspace=0.18)
    paths = _save(fig, figures_dir / f"{lane}_meep_near_field", formats, dpi)
    plt.close(fig)
    return paths


def _plot_far_field(
    records: Sequence[Mapping[str, Any]], *, lane: str, figures_dir: Path, formats: Sequence[str], dpi: int
) -> list[Path]:
    plt = _configure_matplotlib()
    rows, cols = _grid(len(records))
    fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, 220 / 25.4), subplot_kw={"projection": "polar"}, squeeze=False)
    for position, ax in enumerate(axes.flat):
        if position >= len(records):
            ax.axis("off")
            continue
        row = records[position]
        result = row["result"]
        angles = np.r_[result.far_angles_rad, result.far_angles_rad[0] + 2.0 * np.pi]
        radial = np.r_[row["far_normalized"], row["far_normalized"][0]]
        ax.plot(angles, radial, color="#225ea8", linewidth=0.9)
        ax.fill(angles, radial, color="#9ecae1", alpha=0.45)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_rlim(0.0, 1.0)
        ax.set_rticks([0.5, 1.0])
        ax.set_thetagrids([0, 90, 180, 270], labels=["+x", "+y", "−x", "−y"])
        ax.tick_params(labelsize=5.0)
        ax.set_title(f"{row['sample_label']} · f={result.frequency:.3f}", pad=8)
    fig.suptitle(
        f"{lane.replace('_', ' ')} · normalized angular far-field radial Poynting flux",
        fontsize=8,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.035, top=0.91, wspace=0.55, hspace=0.62)
    paths = _save(fig, figures_dir / f"{lane}_meep_far_field", formats, dpi)
    plt.close(fig)
    return paths


def _plot_integrated_response(
    records: Sequence[Mapping[str, Any]],
    *,
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
    arrow_count: int,
    arrow_threshold: float,
) -> list[Path]:
    """Place both material lanes and both response views in one comparison plate."""

    plt = _configure_matplotlib()
    by_lane = {lane: [row for row in records if row["lane"] == lane] for lane in ("graded_medium", "binary_air_holes")}
    graded, binary = by_lane["graded_medium"], by_lane["binary_air_holes"]
    if len(graded) != len(binary) or any(g["sample_label"] != b["sample_label"] for g, b in zip(graded, binary)):
        raise ValueError("integrated response figure requires aligned sample labels across both lanes")
    sample_rows = int(np.ceil(len(graded) / 2))
    fig = plt.figure(figsize=(366 / 25.4, 366 / 25.4))
    grid = fig.add_gridspec(sample_rows, 8, wspace=0.42, hspace=0.34)
    for index, (graded_row, binary_row) in enumerate(zip(graded, binary)):
        row_index, column_group = divmod(index, 2)
        for panel_offset, (record, view, lane_short) in enumerate(
            (
                (graded_row, "near", "G"),
                (graded_row, "far", "G"),
                (binary_row, "near", "B"),
                (binary_row, "far", "B"),
            )
        ):
            column_index = 4 * column_group + panel_offset
            result = record["result"]
            if view == "near":
                ax = fig.add_subplot(grid[row_index, column_index])
                extent = (result.x[0], result.x[-1], result.y[0], result.y[-1])
                ax.imshow(record["near_normalized"], origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
                ax.contour(result.x, result.y, result.epsilon, levels=[0.5 * (1.0 + 8.9)], colors="#7FDBFF", linewidths=0.30, alpha=0.75)
                qx, qy, qu, qv = poynting_quiver_samples(
                    result.x,
                    result.y,
                    result.poynting_x,
                    result.poynting_y,
                    target_count=max(5, arrow_count - 3),
                    relative_threshold=arrow_threshold,
                )
                ax.quiver(qx, qy, qu, qv, color="white", pivot="mid", angles="xy", scale_units="xy", scale=1.9, width=0.006)
                ax.annotate("", xy=(0.20, 0.91), xytext=(0.04, 0.91), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#00d4ff", "lw": 0.8})
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax = fig.add_subplot(grid[row_index, column_index], projection="polar")
                angles = np.r_[result.far_angles_rad, result.far_angles_rad[0] + 2.0 * np.pi]
                radial = np.r_[record["far_normalized"], record["far_normalized"][0]]
                ax.plot(angles, radial, color="#225ea8", linewidth=0.75)
                ax.fill(angles, radial, color="#9ecae1", alpha=0.45)
                ax.set_theta_zero_location("E")
                ax.set_theta_direction(1)
                ax.set_rlim(0.0, 1.0)
                ax.set_rticks([1.0])
                ax.set_thetagrids([0, 90, 180, 270], labels=["+x", "+y", "−x", "−y"])
                ax.tick_params(labelsize=4.0, pad=1)
            ax.set_title(f"{record['sample_label']} {lane_short}-{view} · f={result.frequency:.3f}", fontsize=5.0, pad=4)
    fig.suptitle(
        "Meep finite-array responses · G=graded · B=binary · near: energy + Poynting arrows · far: radial flux",
        fontsize=8,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.018, right=0.99, bottom=0.018, top=0.95)
    paths = _save(fig, figures_dir / "full_cell_dual_lane_meep_response_overview", formats, dpi)
    plt.close(fig)
    return paths


def _write_hdf5(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    import h5py

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "phc2d.meep_scattering_response.v1"
        handle.create_dataset("sample_label", data=np.asarray([row["sample_label"] for row in records], dtype=object), dtype=string_dtype)
        handle.create_dataset("sample_id", data=np.asarray([row["sample_id"] for row in records], dtype=object), dtype=string_dtype)
        handle.create_dataset("lane", data=np.asarray([row["lane"] for row in records], dtype=object), dtype=string_dtype)
        handle.create_dataset("frequency", data=np.asarray([row["result"].frequency for row in records], dtype=np.float64))
        for name in ("epsilon", "electric_energy", "poynting_x", "poynting_y"):
            handle.create_dataset(name, data=np.stack([getattr(row["result"], name) for row in records]), compression="gzip")
        handle.create_dataset("near_normalized", data=np.stack([row["near_normalized"] for row in records]), compression="gzip")
        handle.create_dataset("x", data=np.stack([row["result"].x for row in records]))
        handle.create_dataset("y", data=np.stack([row["result"].y for row in records]))
        handle.create_dataset("far_angles_rad", data=np.stack([row["result"].far_angles_rad for row in records]))
        handle.create_dataset("far_radial_flux", data=np.stack([row["result"].far_radial_flux for row in records]))
        handle.create_dataset("far_normalized", data=np.stack([row["far_normalized"] for row in records]))
        handle.create_dataset(
            "solver_metadata_json",
            data=np.asarray([json.dumps(_jsonable(row["result"].metadata), sort_keys=True) for row in records], dtype=object),
            dtype=string_dtype,
        )


def _write_response_shard(path: Path, result: MeepScatteringResult) -> None:
    np.savez_compressed(
        path,
        frequency=np.asarray(result.frequency),
        x=result.x,
        y=result.y,
        epsilon=result.epsilon,
        electric_energy=result.electric_energy,
        poynting_x=result.poynting_x,
        poynting_y=result.poynting_y,
        far_angles_rad=result.far_angles_rad,
        far_radial_flux=result.far_radial_flux,
        runtime_s=np.asarray(result.runtime_s),
        metadata_json=np.asarray(json.dumps(_jsonable(result.metadata), sort_keys=True)),
    )


def _read_response_shard(
    path: Path,
    *,
    frequency: float,
    geometry_hash: str,
    response_config: Mapping[str, Any],
) -> MeepScatteringResult:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        cached_frequency = float(payload["frequency"].item())
        if not np.isclose(cached_frequency, float(frequency), rtol=0.0, atol=1.0e-10):
            raise RuntimeError(f"response shard frequency mismatch: {cached_frequency} != {frequency}")
        if metadata.get("geometry_hash") != geometry_hash:
            raise RuntimeError("response shard geometry hash mismatch")
        if metadata.get("response_config") != dict(response_config):
            raise RuntimeError("response shard solver-config mismatch")
        return MeepScatteringResult(
            frequency=cached_frequency,
            x=np.asarray(payload["x"]),
            y=np.asarray(payload["y"]),
            epsilon=np.asarray(payload["epsilon"]),
            electric_energy=np.asarray(payload["electric_energy"]),
            poynting_x=np.asarray(payload["poynting_x"]),
            poynting_y=np.asarray(payload["poynting_y"]),
            far_angles_rad=np.asarray(payload["far_angles_rad"]),
            far_radial_flux=np.asarray(payload["far_radial_flux"]),
            runtime_s=float(payload["runtime_s"].item()),
            metadata=metadata,
        )


def build_full_cell_scattering_gallery(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    max_samples_per_dataset: int | None = None,
    solver_fn: ScatteringSolverFn | None = None,
    skip_environment_check: bool = False,
) -> Path:
    """Run finite-array Meep responses for the frozen selected sample lineage."""

    started = time.perf_counter()
    config_file = _repo_path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not skip_environment_check:
        is_wsl = "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
        if not is_wsl or os.environ.get("CONDA_DEFAULT_ENV") != "meep_env":
            raise RuntimeError("Meep scattering requires WSL Ubuntu with conda environment meep_env")
    if solver_fn is None:
        from phc2d.solvers import solve_finite_array_scattering_meep

        solver_fn = solve_finite_array_scattering_meep

    output_cfg = dict(raw["output"])
    experiment = dict(raw["experiment"])
    out = make_run_dir(str(experiment["experiment_id"]), runs_dir=_repo_path(output_cfg.get("runs_dir", "runs")), run_dir=run_dir)
    figures_dir = ensure_figures_dir(out)
    shards_dir = out / "response_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=out / "run.log", run_meta={"stage": "meep_scattering_response"})
    write_resolved_config(out, yaml.safe_dump({**raw, "runtime_overrides": {"max_samples_per_dataset": max_samples_per_dataset}}, sort_keys=False, allow_unicode=True))
    write_run_meta(out, {"stage": "meep_scattering_response", "primary_layer": "L3", "config_path": _relative(config_file), "validation_boundary": "wsl_meep_env_real_meep_finite_array_scattering"})

    source_run = _repo_path(raw["source_gallery_run"])
    with (source_run / "selected_samples.csv").open(encoding="utf-8", newline="") as handle:
        selected = list(csv.DictReader(handle))
    response_cfg = MeepScatteringConfig.from_mapping(dict(raw["response"]))
    gap_range = tuple(int(value) for value in raw["bands"]["gap_lower_bands_one_based"])
    structures_by_lane: dict[str, tuple[FullCellStructureSpec, ...]] = {}
    bands_by_lane: dict[str, np.ndarray] = {}
    dataset_hashes: dict[str, str] = {}
    import h5py

    for dataset in raw["datasets"]:
        lane, path = str(dataset["lane"]), _repo_path(dataset["path"])
        with h5py.File(path, "r") as handle:
            structures_by_lane[lane] = tuple(FullCellStructureSpec.from_mapping(json.loads(_decode(value))) for value in handle["structure_json"][:])
            bands_by_lane[lane] = np.asarray(handle["Wh"], dtype=np.float32)
        dataset_hashes[lane] = _hash(path)

    records: list[dict[str, Any]] = []
    runtimes: list[float] = []
    reused_shard_count = 0
    executed_sample_count = 0
    for lane in structures_by_lane:
        lane_rows = [row for row in selected if row["lane"] == lane]
        if max_samples_per_dataset is not None:
            lane_rows = lane_rows[: int(max_samples_per_dataset)]
        for row in lane_rows:
            index = int(row["dataset_index"])
            gap = select_largest_gap_edge(
                bands_by_lane[lane][index],
                lower_band_min_one_based=gap_range[0],
                lower_band_max_one_based=gap_range[1],
            )
            structure = structures_by_lane[lane][index]
            shard_path = shards_dir / f"{lane}_{row['sample_label']}.npz"
            if shard_path.is_file():
                result = _read_response_shard(
                    shard_path,
                    frequency=gap.frequency,
                    geometry_hash=structure.geometry_hash,
                    response_config=response_cfg.__dict__,
                )
                reused_shard_count += 1
                logger.info("Reused response shard lane=%s label=%s path=%s", lane, row["sample_label"], shard_path)
            else:
                result = solver_fn(structure, frequency=gap.frequency, config=response_cfg)
                _write_response_shard(shard_path, result)
                executed_sample_count += 1
            near_normalized, far_normalized = normalize_response_for_display(result.electric_energy, result.far_radial_flux)
            record = {"lane": lane, "sample_label": row["sample_label"], "sample_id": row["sample_id"], "dataset_index": index, "gap": gap, "result": result, "near_normalized": near_normalized, "far_normalized": far_normalized}
            records.append(record)
            runtimes.append(result.runtime_s)
            logger.info("Completed Meep response lane=%s label=%s sample=%s frequency=%.9g runtime=%.3fs", lane, row["sample_label"], row["sample_id"], gap.frequency, result.runtime_s)

    _write_hdf5(out / "scattering_response_records.h5", records)
    with (out / "response_records.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["lane", "sample_label", "sample_id", "dataset_index", "frequency", "gap", "gap_status", "runtime_s", "farfield_negative_flux_fraction_before_clipping"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"lane": row["lane"], "sample_label": row["sample_label"], "sample_id": row["sample_id"], "dataset_index": row["dataset_index"], "frequency": row["result"].frequency, "gap": row["gap"].gap, "gap_status": row["gap"].status, "runtime_s": row["result"].runtime_s, "farfield_negative_flux_fraction_before_clipping": row["result"].metadata["farfield_negative_flux_fraction_before_clipping"]})

    formats = [str(value) for value in output_cfg["formats"]]
    dpi = int(output_cfg["dpi"])
    figure_paths: list[Path] = []
    for lane in structures_by_lane:
        lane_records = [row for row in records if row["lane"] == lane]
        figure_paths += _plot_near_field(lane_records, lane=lane, figures_dir=figures_dir, formats=formats, dpi=dpi, arrow_count=int(output_cfg["arrow_count"]), arrow_threshold=float(output_cfg["arrow_relative_threshold"]))
        figure_paths += _plot_far_field(lane_records, lane=lane, figures_dir=figures_dir, formats=formats, dpi=dpi)
    figure_paths += _plot_integrated_response(
        records,
        figures_dir=figures_dir,
        formats=formats,
        dpi=dpi,
        arrow_count=int(output_cfg["arrow_count"]),
        arrow_threshold=float(output_cfg["arrow_relative_threshold"]),
    )

    manifest = {
        "schema_version": "phc2d.meep_scattering_gallery_manifest.v1",
        "primary_layer": "L3",
        "core_conclusion": "A finite tiled full-cell structure produces a driven TE near-field energy-flow pattern and angular far-field scattering response at its selected band-edge frequency.",
        "source_gallery_run": _relative(source_run),
        "dataset_sha256": dataset_hashes,
        "response_contract": {"geometry": "finite_tiled_array", "source": "normally_incident_+x_TE_Hz_Gaussian_pulse", "near_field": "normalized_electric_energy_with_time_averaged_Poynting_arrows", "far_field": "closed_surface_near2far_outward_radial_Poynting", "normalization": "per_sample_visual_only"},
        "figures": [{"path": _relative(path), "sha256": _hash(path), "format": path.suffix.lstrip(".")} for path in figure_paths],
        "evidence_boundary": "finite_array_response_under_frozen_assumptions_not_intrinsic_infinite_crystal_property",
    }
    write_json(out / "figure_manifest.json", manifest)
    elapsed = time.perf_counter() - started
    metrics = {"wall_time_total_s": elapsed, "meep_scattering_time_s": float(sum(runtimes)), "meep_time_per_sample_s": float(np.mean(runtimes)), "meep_time_max_sample_s": float(np.max(runtimes)), "throughput_samples_per_s": len(records) / max(sum(runtimes), 1e-12), "sample_count_total": len(records), "meep_executed_this_invocation_count": executed_sample_count, "response_shard_reuse_count": reused_shard_count, "figure_file_count": len(figure_paths)}
    write_metrics(out, metrics)
    with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "wall_time_s", "record_count"])
        writer.writeheader()
        writer.writerow({"phase": "real_meep_scattering", "wall_time_s": sum(runtimes), "record_count": len(records)})
        writer.writerow({"phase": "total", "wall_time_s": elapsed, "record_count": len(figure_paths)})
    write_results(out, {"status": "full_cell_meep_scattering_complete", "sample_count_total": len(records), "response_shard_count": len(list(shards_dir.glob("*.npz"))), "meep_executed_this_invocation_count": executed_sample_count, "response_shard_reuse_count": reused_shard_count, "figure_file_count": len(figure_paths), "near_field_quantity": "normalized_electric_energy_with_time_averaged_Poynting_arrows", "far_field_quantity": "normalized_outward_radial_Poynting_flux", "validation_boundary": "wsl_meep_env_real_meep_finite_array_scattering", "source_gallery_acceptance_advanced": False})
    return out


__all__ = ["build_full_cell_scattering_gallery", "normalize_response_for_display", "poynting_quiver_samples"]
