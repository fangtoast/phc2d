"""Traceable full-cell structure, band, and real-MPB eigenfield galleries.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.field_eigen import canonicalize_eigenfields
from phc2d.data.full_cell import FullCellStructureSpec
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
from phc2d.solvers.mpb import MpbEigenfieldResult
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


ROOT = Path(__file__).resolve().parents[3]
EigenfieldSolverFn = Callable[..., MpbEigenfieldResult]


@dataclass(frozen=True)
class GalleryDataset:
    """The minimum full-cell arrays needed by the gallery runner."""

    path: Path
    lane: str
    status: str
    structures: tuple[FullCellStructureSpec, ...]
    Wh: np.ndarray
    k_points: np.ndarray
    k_grid_index: np.ndarray
    idx_test: np.ndarray
    epsilon_128: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GapEdgeSelection:
    """One complete- or pseudo-gap upper-edge mode selected from a full BZ."""

    lower_band_one_based: int
    upper_band_one_based: int
    gap: float
    status: str
    k_index: int
    frequency: float


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_full_cell_stratum(structure: FullCellStructureSpec, lane: str) -> str:
    """Return the frozen generator stratum for one full-cell structure."""

    if lane == "graded_medium":
        if structure.generator_family != "graded_rbf_fourier":
            raise ValueError(f"graded_medium requires graded_rbf_fourier, got {structure.generator_family}")
        count = int(structure.metadata.get("core_count", len(structure.parameters.get("blobs", []))))
        if count not in {1, 2}:
            raise ValueError(f"graded structure must contain one or two cores, got {count}")
        return f"graded_core_{count}"
    if lane == "binary_air_holes":
        if structure.generator_family == "binary_spectral_threshold":
            return "spectral_threshold"
        if structure.generator_family == "binary_polar_holes":
            count = int(structure.parameters.get("hole_count", len(structure.parameters.get("holes", []))))
            if count not in {1, 2}:
                raise ValueError(f"binary polar structure must contain one or two holes, got {count}")
            return f"polar_hole_{count}"
        raise ValueError(f"unsupported binary generator family: {structure.generator_family}")
    raise ValueError(f"unsupported gallery lane: {lane}")


def select_stratified_test_indices(
    structures: Sequence[FullCellStructureSpec],
    idx_test: Sequence[int],
    *,
    lane: str,
    quotas: Mapping[str, int],
    seed: int,
) -> list[int]:
    """Select a reproducible, quota-balanced sample from the test split only."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for raw_index in np.asarray(idx_test, dtype=np.int64).reshape(-1):
        index = int(raw_index)
        grouped[infer_full_cell_stratum(structures[index], lane)].append(index)
    if set(grouped) != set(quotas):
        raise ValueError(f"{lane} test strata mismatch: expected {sorted(quotas)}, got {sorted(grouped)}")
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for stratum in quotas:
        count = int(quotas[stratum])
        candidates = np.asarray(sorted(grouped[stratum]), dtype=np.int64)
        if count <= 0 or count > len(candidates):
            raise ValueError(f"invalid quota {count} for {stratum} with {len(candidates)} test samples")
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.extend(int(value) for value in np.sort(chosen))
    return selected


def sampled_gxm_path(k_grid_index: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    """Return the exact 13-point Gamma-X9-M9-Gamma path from a square 9x9 grid."""

    indices = np.asarray(k_grid_index, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("k_grid_index must have shape [n_k_points, 2]")
    lookup = {tuple(int(value) for value in row): pos for pos, row in enumerate(indices)}
    edge = max(value[0] for value in lookup)
    if edge != 4 or min(value[0] for value in lookup) != -4:
        raise ValueError("sampled Gamma-X9-M9-Gamma path requires the frozen 9x9 full-BZ index grid")
    targets = (
        [(value, 0) for value in range(0, edge + 1)]
        + [(edge, value) for value in range(1, edge + 1)]
        + [(value, value) for value in range(edge - 1, -1, -1)]
    )
    try:
        path = np.asarray([lookup[target] for target in targets], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"full-BZ grid is missing sampled path coordinate {exc.args[0]}") from exc
    return path, (0, edge, 2 * edge, 3 * edge), (r"$\Gamma$", r"$X_9$", r"$M_9$", r"$\Gamma$")


def select_largest_gap_edge(
    Wh: np.ndarray,
    *,
    lower_band_min_one_based: int = 2,
    lower_band_max_one_based: int | None = None,
) -> GapEdgeSelection:
    """Select the upper edge of the largest configured full-BZ adjacent-band gap."""

    bands = np.asarray(Wh, dtype=np.float64)
    if bands.ndim != 2 or bands.shape[1] < 2 or bands.shape[0] < 1:
        raise ValueError(f"Wh must have shape [n_k_points, n_bands>=2], got {bands.shape}")
    if np.any(~np.isfinite(bands)):
        raise ValueError("Wh contains non-finite frequencies")
    lower_min = int(lower_band_min_one_based)
    lower_max = int(lower_band_max_one_based) if lower_band_max_one_based is not None else bands.shape[1] - 1
    if lower_min < 1 or lower_max < lower_min or lower_max >= bands.shape[1]:
        raise ValueError(
            f"invalid lower-band range {lower_min}..{lower_max} for {bands.shape[1]} retained bands"
        )
    candidates: list[GapEdgeSelection] = []
    for lower_band in range(lower_min, lower_max + 1):
        lower_index = lower_band - 1
        upper_index = lower_index + 1
        lower_max = float(np.max(bands[:, lower_index]))
        k_index = int(np.argmin(bands[:, upper_index]))
        upper_min = float(bands[k_index, upper_index])
        gap = upper_min - lower_max
        candidates.append(
            GapEdgeSelection(
                lower_band_one_based=lower_band,
                upper_band_one_based=lower_band + 1,
                gap=gap,
                status="complete_gap" if gap > 0.0 else "pseudo_gap",
                k_index=k_index,
                frequency=upper_min,
            )
        )
    return max(candidates, key=lambda row: (row.gap, -row.lower_band_one_based))


def normalized_electric_energy_density(
    efield: np.ndarray,
    hfield: np.ndarray,
    *,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return canonical E/H fields and per-sample-normalized epsilon times |E| squared."""

    e_normalized, h_normalized, normalization = canonicalize_eigenfields(
        efield,
        hfield,
        epsilon=np.asarray(epsilon, dtype=np.float64),
        domain_extent=(1.0, 1.0),
    )
    nx, ny = int(e_normalized.shape[0]), int(e_normalized.shape[1])
    eps = np.asarray(epsilon, dtype=np.float64)
    if nx % eps.shape[0] or ny % eps.shape[1]:
        raise ValueError(f"epsilon grid {eps.shape} cannot be lifted to field grid {(nx, ny)}")
    eps_field = np.repeat(np.repeat(eps, nx // eps.shape[0], axis=0), ny // eps.shape[1], axis=1)
    axes = tuple(range(2, e_normalized.ndim))
    magnitude = np.sum(np.abs(e_normalized) ** 2, axis=axes) if axes else np.abs(e_normalized) ** 2
    energy = eps_field * np.asarray(magnitude, dtype=np.float64)
    maximum = float(np.max(energy))
    if not np.isfinite(maximum) or maximum <= 0.0 or np.any(~np.isfinite(energy)):
        raise ValueError("electric energy density is zero or non-finite")
    normalized = (energy / maximum).astype(np.float32)
    return e_normalized, h_normalized, normalized, {
        **normalization,
        "visual_quantity": "epsilon_times_sum_abs_E_components_squared",
        "visual_normalization": "per_sample_max_equals_one",
        "energy_density_max_before_visual_normalization": maximum,
    }


def _load_gallery_dataset(path: Path, *, lane: str, status: str) -> GalleryDataset:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise ImportError("h5py is required for full-cell gallery generation") from exc

    with h5py.File(path, "r") as handle:
        structures = tuple(
            FullCellStructureSpec.from_mapping(json.loads(_decode(value))) for value in handle["structure_json"][:]
        )
        metadata = json.loads(_decode(handle.attrs.get("metadata_json", "{}")))
        return GalleryDataset(
            path=path,
            lane=lane,
            status=status,
            structures=structures,
            Wh=np.asarray(handle["Wh"], dtype=np.float32),
            k_points=np.asarray(handle["k_points"], dtype=np.float64),
            k_grid_index=np.asarray(handle["k_grid_index"], dtype=np.int64),
            idx_test=np.asarray(handle["idx_test"], dtype=np.int64),
            epsilon_128=np.empty((0, 128, 128), dtype=np.float32),
            metadata=metadata,
        )


def _read_selected_epsilon(path: Path, indices: Sequence[int]) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        if "epsilon_grid_128" not in handle:
            raise ValueError(f"dataset lacks epsilon_grid_128: {path}")
        return np.stack([np.asarray(handle["epsilon_grid_128"][int(index)], dtype=np.float32) for index in indices])


def _validate_python_runtime() -> None:
    try:
        import h5py  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Python gallery dependency is unavailable: {exc}") from exc


def _validate_wsl_mpb_boundary() -> None:
    is_wsl = "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))
    if not is_wsl or os.environ.get("CONDA_DEFAULT_ENV") != "meep_env":
        raise RuntimeError("Real-MPB gallery generation requires WSL Ubuntu with conda environment meep_env")


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"gallery config must be a mapping: {path}")
    return dict(raw)


def _validate_config(raw: Mapping[str, Any]) -> None:
    datasets = list(raw.get("datasets") or [])
    if len(datasets) != 2:
        raise ValueError("gallery config must define exactly two datasets")
    sampling = dict(raw.get("sampling") or {})
    sample_count = int(sampling.get("sample_count_per_dataset", 0))
    if sampling.get("split") != "test" or sample_count <= 0:
        raise ValueError("gallery sampling contract requires a positive sample count from the test split")
    for dataset in datasets:
        quota_total = sum(int(value) for value in dict(dataset.get("quotas") or {}).values())
        if quota_total != sample_count:
            raise ValueError(f"dataset quotas must sum to sample_count_per_dataset={sample_count}, got {quota_total}")
    band_cfg = dict(raw.get("bands") or {})
    gap_range = tuple(int(value) for value in band_cfg.get("gap_lower_bands_one_based", ()))
    if len(gap_range) != 2 or gap_range[0] < 1 or gap_range[1] < gap_range[0]:
        raise ValueError("bands.gap_lower_bands_one_based must contain an inclusive positive range")
    field = dict(raw.get("field") or {})
    if field.get("quantity") != "epsilon_abs_e_squared" or field.get("normalization") != "per_sample_max":
        raise ValueError("gallery field contract requires per-sample-normalized epsilon_abs_e_squared")
    output = dict(raw.get("output") or {})
    formats = tuple(str(item) for item in output.get("formats", ()))
    if formats != ("png", "svg", "pdf"):
        raise ValueError("gallery output formats must be [png, svg, pdf]")


def _solver_config(raw: Mapping[str, Any]) -> FullCellMpbBandSolverConfig:
    solver = dict(raw.get("solver") or {})
    return FullCellMpbBandSolverConfig(
        epsilon_air=float(solver["epsilon_air"]),
        epsilon_material=float(solver["epsilon_material"]),
        resolution=int(solver["resolution"]),
        polarization=str(solver["polarization"]),
        num_bands=int(solver["num_bands"]),
        band_guard_count=int(solver["band_guard_count"]),
        band_order_policy=str(solver["band_order_policy"]),
        eigensolver_tolerance=float(solver["eigensolver_tolerance"]),
        mesh_size=int(solver["mesh_size"]),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_near_field_hdf5(path: Path, records: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    import h5py

    if not records:
        raise ValueError("near-field record list is empty")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "phc2d.full_cell_gallery_field.v1"
        handle.attrs["metadata_json"] = json.dumps(_jsonable(metadata), ensure_ascii=False, sort_keys=True)
        handle.create_dataset("lane", data=np.asarray([row["lane"] for row in records], dtype=object), dtype=string_dtype)
        handle.create_dataset("sample_label", data=np.asarray([row["sample_label"] for row in records], dtype=object), dtype=string_dtype)
        handle.create_dataset("sample_id", data=np.asarray([row["sample_id"] for row in records], dtype=object), dtype=string_dtype)
        handle.create_dataset("dataset_index", data=np.asarray([row["dataset_index"] for row in records], dtype=np.int64))
        handle.create_dataset("k_points", data=np.stack([row["k_point"] for row in records]).astype(np.float64))
        handle.create_dataset("band_index", data=np.asarray([row["band_index"] for row in records], dtype=np.int32))
        handle.create_dataset("dataset_frequency", data=np.asarray([row["dataset_frequency"] for row in records], dtype=np.float64))
        handle.create_dataset("replay_frequency", data=np.asarray([row["replay_frequency"] for row in records], dtype=np.float64))
        handle.create_dataset("frequency_mre", data=np.asarray([row["frequency_mre"] for row in records], dtype=np.float64))
        handle.create_dataset("epsilon_128", data=np.stack([row["epsilon_128"] for row in records]).astype(np.float32), compression="gzip")
        handle.create_dataset("efield_complex", data=np.stack([row["efield"] for row in records]).astype(np.complex64), compression="gzip")
        handle.create_dataset("hfield_complex", data=np.stack([row["hfield"] for row in records]).astype(np.complex64), compression="gzip")
        handle.create_dataset("electric_energy_density_normalized", data=np.stack([row["energy"] for row in records]).astype(np.float32), compression="gzip")
        handle.create_dataset(
            "normalization_json",
            data=np.asarray([json.dumps(_jsonable(row["normalization"]), ensure_ascii=False, sort_keys=True) for row in records], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "solver_provenance_json",
            data=np.asarray([json.dumps(_jsonable(row["solver_metadata"]), ensure_ascii=False, sort_keys=True) for row in records], dtype=object),
            dtype=string_dtype,
        )


def _figure_grid(count: int) -> tuple[int, int]:
    square = int(math.isqrt(count))
    if square * square == count:
        return square, square
    cols = min(4, max(1, int(math.ceil(math.sqrt(count)))))
    return int(math.ceil(count / cols)), cols


def _save_figure(fig: Any, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    paths: list[Path] = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        kwargs = {"dpi": int(dpi)} if suffix == "png" else {}
        # Preserve the configured physical canvas size. ``bbox_inches="tight"``
        # would trim each plate by a different amount and break the frozen
        # 183 mm double-column-width contract.
        fig.savefig(path, facecolor="white", **kwargs)
        paths.append(path)
    return paths


def _configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    logging.getLogger("fontTools").setLevel(logging.WARNING)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )
    return plt


def _medium_colormap() -> Any:
    """Map air to white and increasing permittivity to progressively darker blue."""

    import matplotlib as mpl

    return mpl.colors.LinearSegmentedColormap.from_list(
        "air_white_material_blue",
        ["#ffffff", "#deebf7", "#9ecae1", "#4292c6", "#084594"],
    )


def _plot_medium_plate(
    records: Sequence[Mapping[str, Any]], *, lane: str, figures_dir: Path, formats: Sequence[str], dpi: int
) -> list[Path]:
    plt = _configure_matplotlib()
    rows, cols = _figure_grid(len(records))
    fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, 183 / 25.4), squeeze=False)
    image = None
    for pos, ax in enumerate(axes.flat):
        if pos >= len(records):
            ax.axis("off")
            continue
        row = records[pos]
        image = ax.imshow(
            row["epsilon_128"],
            origin="lower",
            cmap=_medium_colormap(),
            vmin=1.0,
            vmax=8.9,
            interpolation="nearest",
        )
        short_stratum = str(row["stratum"]).replace("graded_", "").replace("polar_", "").replace("_", " ")
        ax.set_title(f"{row['sample_label']} · {short_stratum}", fontsize=5.7, pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        colorbar_ax = fig.add_axes([0.925, 0.25, 0.016, 0.50])
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        colorbar.set_label(r"Relative permittivity $\varepsilon_r$")
    fig.suptitle(f"{lane.replace('_', ' ')} · test-split dielectric maps (n={len(records)})", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.03, top=0.94, wspace=0.10, hspace=0.17)
    paths = _save_figure(fig, figures_dir / f"{lane}_medium_4x4", formats, dpi)
    plt.close(fig)
    return paths


def _plot_band_plate(
    records: Sequence[Mapping[str, Any]],
    *,
    lane: str,
    path_indices: np.ndarray,
    ticks: Sequence[int],
    tick_labels: Sequence[str],
    frequency_limits: tuple[float, float],
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    plt = _configure_matplotlib()
    rows, cols = _figure_grid(len(records))
    fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, 155 / 25.4), squeeze=False, sharex=True, sharey=True)
    n_bands = int(np.asarray(records[0]["Wh"]).shape[1])
    colors = plt.cm.Blues(np.linspace(0.35, 0.95, n_bands))
    x = np.arange(len(path_indices))
    for pos, ax in enumerate(axes.flat):
        if pos >= len(records):
            ax.axis("off")
            continue
        row = records[pos]
        path_bands = np.asarray(row["Wh"])[path_indices]
        for band_index in range(path_bands.shape[1]):
            ax.plot(x, path_bands[:, band_index], color=colors[band_index], linewidth=0.65)
        ax.axhline(float(row["gap_edge"].frequency), color="#a23b72", linewidth=0.65, linestyle=(0, (2, 1)))
        ax.set_title(
            f"{row['sample_label']} · b{row['gap_edge'].upper_band_one_based} edge · "
            f"Δ={row['gap_edge'].gap:.3f}"
        )
        ax.set_xticks(ticks, tick_labels)
        ax.set_xlim(0, len(path_indices) - 1)
        ax.set_ylim(*frequency_limits)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.4)
        if pos % cols == 0:
            ax.set_ylabel(r"Frequency $a/\lambda$")
    fig.suptitle(
        f"{lane.replace('_', ' ')} · sampled TE bands "
        r"$\Gamma$–$X_9$–$M_9$–$\Gamma$ "
        r"($X_9$=(4/9,0), $M_9$=(4/9,4/9); no interpolation)",
        fontsize=8,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.07, top=0.92, wspace=0.12, hspace=0.30)
    paths = _save_figure(fig, figures_dir / f"{lane}_bands_4x4", formats, dpi)
    plt.close(fig)
    return paths


def _plot_field_plate(
    records: Sequence[Mapping[str, Any]], *, lane: str, figures_dir: Path, formats: Sequence[str], dpi: int
) -> list[Path]:
    plt = _configure_matplotlib()
    rows, cols = _figure_grid(len(records))
    fig, axes = plt.subplots(rows, cols, figsize=(183 / 25.4, 183 / 25.4), squeeze=False)
    image = None
    for pos, ax in enumerate(axes.flat):
        if pos >= len(records):
            ax.axis("off")
            continue
        row = records[pos]
        image = ax.imshow(row["energy"], origin="lower", cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        kx_index, ky_index = (int(round(9.0 * float(value))) for value in row["k_point"])
        gap = row["gap_edge"]
        tag = "gap" if gap.status == "complete_gap" else "pseudo"
        ax.set_title(
            f"{row['sample_label']} · b{gap.upper_band_one_based} · {tag}\n"
            f"k=({kx_index},{ky_index})/9",
            fontsize=5.0,
            pad=2.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        colorbar_ax = fig.add_axes([0.925, 0.25, 0.016, 0.50])
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        colorbar.set_label(r"Normalized electric energy $\varepsilon|E|^2 / \max$")
    fig.suptitle(
        f"{lane.replace('_', ' ')} · real-MPB upper-gap-edge TE eigenfields (n={len(records)})",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(0.5, 0.012, "Each panel is normalized independently; color does not encode absolute cross-sample field strength.", ha="center", fontsize=5.5)
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.04, top=0.93, wspace=0.10, hspace=0.24)
    paths = _save_figure(fig, figures_dir / f"{lane}_near_field_4x4", formats, dpi)
    plt.close(fig)
    return paths


def _plot_integrated_overview(
    records: Sequence[Mapping[str, Any]],
    *,
    lanes: Sequence[str],
    path_indices: np.ndarray,
    ticks: Sequence[int],
    tick_labels: Sequence[str],
    frequency_limits: tuple[float, float],
    figures_dir: Path,
    formats: Sequence[str],
    dpi: int,
    width_mm: float,
    height_mm: float,
) -> list[Path]:
    """Render all six 4x4 plates as one large, traceable overview figure."""

    if len(lanes) != 2:
        raise ValueError("the integrated overview requires exactly two dataset lanes")
    plt = _configure_matplotlib()
    fig = plt.figure(figsize=(float(width_mm) / 25.4, float(height_mm) / 25.4), facecolor="white")
    subfigures = fig.subfigures(3, 2, height_ratios=(183, 155, 183), wspace=0.025, hspace=0.025)
    medium_cmap = _medium_colormap()

    for column, lane in enumerate(lanes):
        lane_records = [row for row in records if row["lane"] == lane]
        rows, cols = _figure_grid(len(lane_records))

        medium_subfigure = subfigures[0, column]
        medium_axes = medium_subfigure.subplots(rows, cols, squeeze=False)
        medium_image = None
        for position, ax in enumerate(medium_axes.flat):
            if position >= len(lane_records):
                ax.axis("off")
                continue
            row = lane_records[position]
            medium_image = ax.imshow(
                row["epsilon_128"],
                origin="lower",
                cmap=medium_cmap,
                vmin=1.0,
                vmax=8.9,
                interpolation="nearest",
            )
            short_stratum = str(row["stratum"]).replace("graded_", "").replace("polar_", "").replace("_", " ")
            ax.set_title(f"{row['sample_label']} · {short_stratum}", fontsize=5.0, pad=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
        if medium_image is not None:
            colorbar = medium_subfigure.colorbar(
                medium_image,
                ax=medium_axes.ravel().tolist(),
                fraction=0.022,
                pad=0.018,
                aspect=35,
            )
            colorbar.set_label(r"Relative permittivity $\varepsilon_r$", fontsize=5.5)
        medium_subfigure.suptitle(
            f"{'a' if column == 0 else 'b'}  {lane.replace('_', ' ')} dielectric maps",
            fontsize=8,
            fontweight="bold",
            x=0.02,
            ha="left",
        )
        medium_subfigure.subplots_adjust(left=0.025, right=0.91, bottom=0.025, top=0.92, wspace=0.10, hspace=0.17)

        band_subfigure = subfigures[1, column]
        band_axes = band_subfigure.subplots(rows, cols, squeeze=False, sharex=True, sharey=True)
        n_bands = int(np.asarray(lane_records[0]["Wh"]).shape[1])
        colors = plt.cm.Blues(np.linspace(0.35, 0.95, n_bands))
        x = np.arange(len(path_indices))
        for position, ax in enumerate(band_axes.flat):
            if position >= len(lane_records):
                ax.axis("off")
                continue
            row = lane_records[position]
            path_bands = np.asarray(row["Wh"])[path_indices]
            for band_index in range(path_bands.shape[1]):
                ax.plot(x, path_bands[:, band_index], color=colors[band_index], linewidth=0.55)
            ax.axhline(float(row["gap_edge"].frequency), color="#a23b72", linewidth=0.6, linestyle=(0, (2, 1)))
            ax.set_title(
                f"{row['sample_label']} · b{row['gap_edge'].upper_band_one_based} · "
                f"Δ={row['gap_edge'].gap:.3f}",
                fontsize=5.0,
                pad=1.5,
            )
            ax.set_xticks(ticks, tick_labels)
            ax.set_xlim(0, len(path_indices) - 1)
            ax.set_ylim(*frequency_limits)
            ax.grid(axis="x", color="#d9d9d9", linewidth=0.35)
            if position % cols == 0:
                ax.set_ylabel(r"Frequency $a/\lambda$", fontsize=5.0)
        band_subfigure.suptitle(
            f"{'c' if column == 0 else 'd'}  {lane.replace('_', ' ')} sampled TE bands",
            fontsize=8,
            fontweight="bold",
            x=0.02,
            ha="left",
        )
        band_subfigure.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.90, wspace=0.12, hspace=0.30)

        field_subfigure = subfigures[2, column]
        field_axes = field_subfigure.subplots(rows, cols, squeeze=False)
        field_image = None
        for position, ax in enumerate(field_axes.flat):
            if position >= len(lane_records):
                ax.axis("off")
                continue
            row = lane_records[position]
            field_image = ax.imshow(
                row["energy"],
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            gap = row["gap_edge"]
            tag = "gap" if gap.status == "complete_gap" else "pseudo"
            kx_index, ky_index = (int(round(9.0 * float(value))) for value in row["k_point"])
            ax.set_title(
                f"{row['sample_label']} · b{gap.upper_band_one_based} · {tag}\n"
                f"k=({kx_index},{ky_index})/9",
                fontsize=4.7,
                pad=1.5,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        if field_image is not None:
            colorbar = field_subfigure.colorbar(
                field_image,
                ax=field_axes.ravel().tolist(),
                fraction=0.022,
                pad=0.018,
                aspect=35,
            )
            colorbar.set_label(r"Normalized $\varepsilon|E|^2 / \max$", fontsize=5.5)
        field_subfigure.suptitle(
            f"{'e' if column == 0 else 'f'}  {lane.replace('_', ' ')} upper-gap-edge TE eigenfields",
            fontsize=8,
            fontweight="bold",
            x=0.02,
            ha="left",
        )
        field_subfigure.subplots_adjust(left=0.025, right=0.91, bottom=0.045, top=0.90, wspace=0.10, hspace=0.24)
        field_subfigure.text(
            0.47,
            0.012,
            "Per-sample normalization; colors do not compare absolute field strength.",
            ha="center",
            fontsize=5.0,
        )

    paths = _save_figure(fig, figures_dir / "full_cell_dual_lane_overview", formats, dpi)
    plt.close(fig)
    return paths


def build_full_cell_sample_gallery(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    max_samples_per_dataset: int | None = None,
    solver_fn: EigenfieldSolverFn | None = None,
    skip_environment_check: bool = False,
) -> Path:
    """Build traceable structure/band/field plates for two full-cell datasets."""

    started = time.perf_counter()
    config_file = _repo_path(config_path)
    raw = _load_config(config_file)
    _validate_config(raw)
    _validate_python_runtime()
    if not skip_environment_check:
        _validate_wsl_mpb_boundary()
    if solver_fn is None:
        from phc2d.solvers import solve_eigenfield_for_full_cell_mpb

        solver_fn = solve_eigenfield_for_full_cell_mpb

    experiment = dict(raw["experiment"])
    output_cfg = dict(raw["output"])
    out = make_run_dir(
        str(experiment["experiment_id"]),
        runs_dir=_repo_path(output_cfg.get("runs_dir", "runs")),
        run_dir=run_dir,
    )
    logger = setup_logging(log_file=out / "run.log", run_meta={"stage": "full_cell_sample_gallery"})
    figures_dir = ensure_figures_dir(out)
    resolved = {**raw, "runtime_overrides": {"max_samples_per_dataset": max_samples_per_dataset}}
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    write_run_meta(
        out,
        {
            "stage": "full_cell_sample_gallery",
            "config_path": _relative(config_file),
            "validation_boundary": "wsl_meep_env_real_mpb_eigenfield_visualization",
        },
    )

    sampling = dict(raw["sampling"])
    band_cfg = dict(raw["bands"])
    gap_lower_min, gap_lower_max = (int(value) for value in band_cfg["gap_lower_bands_one_based"])
    seed = int(sampling["seed"])
    selected_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    datasets: list[GalleryDataset] = []
    dataset_hashes: dict[str, str] = {}
    path_indices: np.ndarray | None = None
    path_ticks: tuple[int, ...] | None = None
    path_labels: tuple[str, ...] | None = None

    load_started = time.perf_counter()
    for dataset_cfg_raw in raw["datasets"]:
        dataset_cfg = dict(dataset_cfg_raw)
        dataset_path = _repo_path(dataset_cfg["path"])
        dataset = _load_gallery_dataset(
            dataset_path,
            lane=str(dataset_cfg["lane"]),
            status=str(dataset_cfg["status"]),
        )
        datasets.append(dataset)
        dataset_hashes[dataset.lane] = _file_hash(dataset_path)
        current_path, current_ticks, current_labels = sampled_gxm_path(dataset.k_grid_index)
        if path_indices is None:
            path_indices, path_ticks, path_labels = current_path, current_ticks, current_labels
        elif not np.array_equal(path_indices, current_path):
            raise ValueError("the two datasets do not share the frozen sampled GXM path")
    load_time_s = time.perf_counter() - load_started

    replay_started = time.perf_counter()
    solver = _solver_config(raw)
    replay_timing: list[float] = []
    max_mre = float(dict(raw["field"])["frequency_mre_max"])
    for dataset, dataset_cfg_raw in zip(datasets, raw["datasets"], strict=True):
        dataset_cfg = dict(dataset_cfg_raw)
        quotas = {str(key): int(value) for key, value in dict(dataset_cfg["quotas"]).items()}
        selected = select_stratified_test_indices(
            dataset.structures,
            dataset.idx_test,
            lane=dataset.lane,
            quotas=quotas,
            seed=seed,
        )
        if max_samples_per_dataset is not None:
            limit = int(max_samples_per_dataset)
            if limit <= 0:
                raise ValueError("max_samples_per_dataset must be positive")
            selected = selected[:limit]
        epsilon_selected = _read_selected_epsilon(dataset.path, selected)
        lane_records: list[dict[str, Any]] = []
        for position, (index, epsilon) in enumerate(zip(selected, epsilon_selected, strict=True), start=1):
            structure = dataset.structures[index]
            sample_label = f"S{position:02d}"
            stratum = infer_full_cell_stratum(structure, dataset.lane)
            gap = select_largest_gap_edge(
                dataset.Wh[index],
                lower_band_min_one_based=gap_lower_min,
                lower_band_max_one_based=gap_lower_max,
            )
            k_point = np.asarray(dataset.k_points[gap.k_index], dtype=np.float64)
            solve_started = time.perf_counter()
            result = solver_fn(
                structure,
                k_point=k_point,
                band_index_one_based=gap.upper_band_one_based,
                config=solver,
            )
            solve_time = time.perf_counter() - solve_started
            replay_timing.append(solve_time)
            denominator = max(abs(float(gap.frequency)), 1.0e-12)
            frequency_mre = abs(float(result.eigenfrequency) - float(gap.frequency)) / denominator
            if not np.isfinite(frequency_mre) or frequency_mre > max_mre:
                raise RuntimeError(
                    f"real-MPB field replay mismatch for {structure.sample_id}: "
                    f"dataset={gap.frequency:.9g}, replay={result.eigenfrequency:.9g}, MRE={frequency_mre:.6%}"
                )
            efield, hfield, energy, normalization = normalized_electric_energy_density(
                result.efield,
                result.hfield,
                epsilon=structure.epsilon_grid(result.efield.shape[:2]),
            )
            stats = dict(structure.metadata.get("stats") or {})
            record = {
                "lane": dataset.lane,
                "dataset_status": dataset.status,
                "sample_label": sample_label,
                "sample_id": structure.sample_id,
                "dataset_index": int(index),
                "split": "test",
                "stratum": stratum,
                "structure": structure,
                "epsilon_128": epsilon,
                "Wh": dataset.Wh[index],
                "gap_edge": gap,
                "k_point": k_point,
                "band_index": gap.upper_band_one_based,
                "dataset_frequency": float(gap.frequency),
                "replay_frequency": float(result.eigenfrequency),
                "frequency_mre": float(frequency_mre),
                "efield": efield,
                "hfield": hfield,
                "energy": energy,
                "normalization": normalization,
                "solver_metadata": result.metadata,
                "solve_time_s": solve_time,
            }
            lane_records.append(record)
            all_records.append(record)
            selected_rows.append(
                {
                    "lane": dataset.lane,
                    "dataset_status": dataset.status,
                    "sample_label": sample_label,
                    "dataset_index": int(index),
                    "sample_id": structure.sample_id,
                    "split": "test",
                    "stratum": stratum,
                    "material_fraction": stats.get("material_fraction"),
                    "gray_fraction": stats.get("gray_fraction"),
                    "air_component_count_periodic": stats.get("air_component_count_periodic"),
                    "geometry_hash": structure.geometry_hash,
                }
            )
            assert path_indices is not None
            for path_position, k_index in enumerate(path_indices):
                for band_index in range(dataset.Wh.shape[2]):
                    band_rows.append(
                        {
                            "lane": dataset.lane,
                            "sample_label": sample_label,
                            "sample_id": structure.sample_id,
                            "dataset_index": int(index),
                            "path_position": int(path_position),
                            "k_index": int(k_index),
                            "k_x": float(dataset.k_points[k_index, 0]),
                            "k_y": float(dataset.k_points[k_index, 1]),
                            "band_index_one_based": band_index + 1,
                            "frequency_a_over_lambda": float(dataset.Wh[index, k_index, band_index]),
                            "selected_gap_lower_band": gap.lower_band_one_based,
                            "selected_gap_upper_band": gap.upper_band_one_based,
                            "selected_gap": gap.gap,
                            "selected_gap_status": gap.status,
                            "selected_gap_k_index": gap.k_index,
                        }
                    )
            logger.info(
                "Completed field replay lane=%s label=%s sample=%s band=%d k=%s MRE=%.3g time=%.3fs",
                dataset.lane,
                sample_label,
                structure.sample_id,
                gap.upper_band_one_based,
                k_point.tolist(),
                frequency_mre,
                solve_time,
            )
        dataset_records = lane_records
        dataset_cfg["_records"] = dataset_records
        dataset.metadata["_gallery_records"] = dataset_records
    replay_time_s = time.perf_counter() - replay_started

    _write_csv(out / "selected_samples.csv", selected_rows)
    _write_csv(out / "band_source.csv", band_rows)
    _write_near_field_hdf5(
        out / "near_field_records.h5",
        all_records,
        {
            "quantity": "epsilon_times_sum_abs_E_components_squared",
            "visual_normalization": "per_sample_max_equals_one",
            "validation_boundary": "wsl_meep_env_real_mpb_eigenfield_visualization",
        },
    )

    render_started = time.perf_counter()
    formats = [str(item) for item in output_cfg["formats"]]
    dpi = int(output_cfg["dpi"])
    figure_paths: list[Path] = []
    global_frequency_min = min(float(np.min(row["Wh"])) for row in all_records)
    global_frequency_max = max(float(np.max(row["Wh"])) for row in all_records)
    padding = 0.02 * max(global_frequency_max - global_frequency_min, 1.0e-3)
    assert path_indices is not None and path_ticks is not None and path_labels is not None
    for dataset in datasets:
        lane_records = [row for row in all_records if row["lane"] == dataset.lane]
        figure_paths.extend(
            _plot_medium_plate(lane_records, lane=dataset.lane, figures_dir=figures_dir, formats=formats, dpi=dpi)
        )
        figure_paths.extend(
            _plot_band_plate(
                lane_records,
                lane=dataset.lane,
                path_indices=path_indices,
                ticks=path_ticks,
                tick_labels=path_labels,
                frequency_limits=(global_frequency_min - padding, global_frequency_max + padding),
                figures_dir=figures_dir,
                formats=formats,
                dpi=dpi,
            )
        )
        figure_paths.extend(
            _plot_field_plate(lane_records, lane=dataset.lane, figures_dir=figures_dir, formats=formats, dpi=dpi)
        )
    figure_paths.extend(
        _plot_integrated_overview(
            all_records,
            lanes=[dataset.lane for dataset in datasets],
            path_indices=path_indices,
            ticks=path_ticks,
            tick_labels=path_labels,
            frequency_limits=(global_frequency_min - padding, global_frequency_max + padding),
            figures_dir=figures_dir,
            formats=formats,
            dpi=dpi,
            width_mm=float(output_cfg.get("integrated_width_mm", 366.0)),
            height_mm=float(output_cfg.get("integrated_height_mm", 521.0)),
        )
    )
    render_time_s = time.perf_counter() - render_started

    manifest = {
        "schema_version": "phc2d.full_cell_gallery_manifest.v1",
        "core_conclusion": "Full-cell dielectric morphology is traceably paired with real-MPB TE bands and upper-gap-edge eigenfield localization.",
        "figure_archetype": "image_plate_plus_quant",
        "backend": "python_matplotlib",
        "sample_count_per_dataset": len(all_records) // len(datasets),
        "sampling": {"split": "test", "seed": seed, "policy": "stratified_random_without_replacement"},
        "band_path": {
            "name": "sampled_Gamma_X9_M9_Gamma",
            "indices": path_indices.tolist(),
            "X9": [4.0 / 9.0, 0.0],
            "M9": [4.0 / 9.0, 4.0 / 9.0],
            "interpolation": False,
        },
        "near_field": {
            "quantity": "epsilon_times_sum_abs_E_components_squared",
            "normalization": "epsilon_weighted_E_l2_then_per_sample_energy_max",
            "cross_sample_absolute_magnitude_comparable": False,
        },
        "medium_color_semantics": {
            "air_epsilon_r_1": "white",
            "material_epsilon_r_8_9": "dark_blue",
            "graded_medium": "continuous_white_to_blue",
        },
        "integrated_overview": {
            "path": "figures/full_cell_dual_lane_overview",
            "layout": "three_response_rows_by_two_dataset_columns",
            "width_mm": float(output_cfg.get("integrated_width_mm", 366.0)),
            "height_mm": float(output_cfg.get("integrated_height_mm", 521.0)),
        },
        "datasets": [
            {"path": _relative(dataset.path), "lane": dataset.lane, "status": dataset.status, "sha256": dataset_hashes[dataset.lane]}
            for dataset in datasets
        ],
        "figures": [
            {"path": _relative(path), "sha256": _file_hash(path), "format": path.suffix.lstrip(".")}
            for path in figure_paths
        ],
        "evidence_boundary": {
            "binary_air_holes": "wsl_cross_environment_acceptance_pass",
            "graded_medium": "cluster_generated_pending_wsl_acceptance",
            "gallery_field_replay": "visualization_only_not_dataset_acceptance",
            "far_field": "not_computed_scattering_contract_absent",
        },
    }
    write_json(out / "figure_manifest.json", manifest)

    wall_time_s = time.perf_counter() - started
    metrics = {
        "wall_time_total_s": wall_time_s,
        "data_load_and_hash_time_s": load_time_s,
        "mpb_replay_time_s": replay_time_s,
        "figure_render_time_s": render_time_s,
        "sample_count_total": len(all_records),
        "mpb_replay_success_count": len(all_records),
        "mpb_replay_failure_count": 0,
        "mpb_replay_throughput_samples_per_s": len(all_records) / replay_time_s if replay_time_s > 0 else 0.0,
        "mpb_replay_time_per_sample_ms": 1000.0 * replay_time_s / len(all_records),
        "mpb_solve_time_mean_s": float(np.mean(replay_timing)),
        "mpb_solve_time_max_s": float(np.max(replay_timing)),
        "frequency_replay_mre_max": float(max(row["frequency_mre"] for row in all_records)),
        "frequency_replay_mre_mean": float(np.mean([row["frequency_mre"] for row in all_records])),
        "figure_file_count": len(figure_paths),
    }
    write_metrics(out, metrics)
    with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "wall_time_s", "record_count"])
        writer.writeheader()
        writer.writerows(
            [
                {"phase": "load_and_hash", "wall_time_s": load_time_s, "record_count": 0},
                {"phase": "real_mpb_field_replay", "wall_time_s": replay_time_s, "record_count": len(all_records)},
                {"phase": "figure_render", "wall_time_s": render_time_s, "record_count": len(figure_paths)},
            ]
        )
    write_results(
        out,
        {
            "status": "full_cell_sample_gallery_complete",
            "sample_count_total": len(all_records),
            "sample_count_per_dataset": len(all_records) // len(datasets),
            "figure_file_count": len(figure_paths),
            "figure_manifest": "figure_manifest.json",
            "near_field_records": "near_field_records.h5",
            "dataset_status": {dataset.lane: dataset.status for dataset in datasets},
            "validation_boundary": "wsl_meep_env_real_mpb_eigenfield_visualization",
            "graded_dataset_acceptance_advanced": False,
            "far_field_computed": False,
        },
    )
    logger.info("Completed full-cell gallery run %s", out)
    return out


__all__ = [
    "GalleryDataset",
    "GapEdgeSelection",
    "build_full_cell_sample_gallery",
    "infer_full_cell_stratum",
    "normalized_electric_energy_density",
    "sampled_gxm_path",
    "select_largest_gap_edge",
    "select_stratified_test_indices",
]
