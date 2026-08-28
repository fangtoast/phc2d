"""Solver and evidence-collection stages for EXP-PHY-005.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from phc2d.data.full_cell import FullCellStructureSpec
from phc2d.physics.effective_index_validation import (
    CANDIDATES,
    candidate_values,
    fit_gamma_slope,
    nearest_gap,
)
from phc2d.physics.effective_medium import effective_refractive_index
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


StructureSolver = Callable[..., tuple[np.ndarray, dict[str, Any]]]
UniformSolver = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def epsilon_from_raster(raster: np.ndarray, solver: Mapping[str, Any]) -> np.ndarray:
    """Convert a frozen material-weight raster to a positive epsilon field."""

    weight = np.asarray(raster, dtype=np.float64)
    if not np.isfinite(weight).all() or np.any(weight < -1.0e-6) or np.any(weight > 1.0 + 1.0e-6):
        raise ValueError("material raster must be finite and lie in [0, 1]")
    epsilon_air = float(solver["epsilon_air"])
    epsilon_material = float(solver["epsilon_material"])
    return epsilon_air + (epsilon_material - epsilon_air) * np.clip(weight, 0.0, 1.0)


def run_uniform_sentinels(
    *,
    k_points: np.ndarray,
    direction_indices: Mapping[str, np.ndarray],
    kappa: np.ndarray,
    config: FullCellMpbBandSolverConfig,
    uniform_solver: UniformSolver,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the three preregistered homogeneous-medium solver controls."""

    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for epsilon in (1.0, 4.0, 8.9):
        bands, metadata = uniform_solver(epsilon, k_points=k_points, config=config)
        provenance[str(epsilon)] = metadata
        fits = {
            direction: fit_gamma_slope(kappa, bands[indices, 0])
            for direction, indices in direction_indices.items()
        }
        expected = float(np.sqrt(epsilon))
        x_index = fits["x_indices"].effective_index
        m_index = fits["m_indices"].effective_index
        rows.append(
            {
                "epsilon": epsilon,
                "expected_index": expected,
                "x_index": x_index,
                "m_index": m_index,
                "x_relative_error": abs(x_index - expected) / expected,
                "m_relative_error": abs(m_index - expected) / expected,
                "direction_difference": abs(x_index - m_index) / (0.5 * (x_index + m_index)),
            }
        )
    return rows, provenance


def collect_slope_evidence(
    contracts: Mapping[str, Any],
    *,
    lanes: Sequence[str],
    k_points: np.ndarray,
    direction_indices: Mapping[str, np.ndarray],
    kappa: np.ndarray,
    solver_config: FullCellMpbBandSolverConfig,
    structure_solver: StructureSolver,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[np.ndarray],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Solve selected structures and collect analytic-versus-MPB slope rows."""

    import h5py

    slope_rows: list[dict[str, Any]] = []
    bands_records: list[np.ndarray] = []
    bands_keys: list[str] = []
    provenance: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for lane in lanes:
        contract = contracts[lane]
        with h5py.File(contract.path, "r") as handle:
            for split, indices in (
                ("train", contract.slope_train),
                ("validation", contract.slope_validation),
            ):
                for index in indices:
                    structure = FullCellStructureSpec.from_mapping(
                        json.loads(_decode(handle["structure_json"][int(index)]))
                    )
                    try:
                        bands, metadata = structure_solver(
                            structure,
                            k_points=k_points,
                            config=solver_config,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "stage": "near_gamma_structure",
                                "lane": lane,
                                "split": split,
                                "dataset_index": int(index),
                                "sample_id": structure.sample_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        continue
                    x_fit = fit_gamma_slope(kappa, bands[direction_indices["x_indices"], 0])
                    m_fit = fit_gamma_slope(kappa, bands[direction_indices["m_indices"], 0])
                    epsilon = epsilon_from_raster(
                        np.asarray(handle[contract.raster_field][int(index)]), contract.solver
                    )
                    old_index = float(np.mean(np.sqrt(epsilon)))
                    te_index = float(effective_refractive_index(epsilon, "TE"))
                    arithmetic_index = float(np.sqrt(np.mean(epsilon)))
                    oracle_mean = 0.5 * (x_fit.effective_index + m_fit.effective_index)
                    row = {
                        "lane": lane,
                        "split": split,
                        "dataset_index": int(index),
                        "sample_id": _decode(handle["sample_id"][int(index)]),
                        "old_index": old_index,
                        "arithmetic_index": arithmetic_index,
                        "te_index": te_index,
                        "x_slope": x_fit.slope,
                        "m_slope": m_fit.slope,
                        "x_inner_slope": x_fit.inner_slope,
                        "m_inner_slope": m_fit.inner_slope,
                        "x_residual_rmse": x_fit.residual_rmse,
                        "m_residual_rmse": m_fit.residual_rmse,
                        "x_index": x_fit.effective_index,
                        "m_index": m_fit.effective_index,
                        "oracle_index_mean": oracle_mean,
                        "x_r2": x_fit.r2_origin,
                        "m_r2": m_fit.r2_origin,
                        "x_window_drift": x_fit.window_drift,
                        "m_window_drift": m_fit.window_drift,
                        "anisotropy": abs(x_fit.effective_index - m_fit.effective_index)
                        / oracle_mean,
                        "old_relative_error": abs(old_index - oracle_mean) / oracle_mean,
                        "arithmetic_relative_error": abs(arithmetic_index - oracle_mean)
                        / oracle_mean,
                        "te_relative_error": abs(te_index - oracle_mean) / oracle_mean,
                    }
                    slope_rows.append(row)
                    key = f"{lane}:{split}:{int(index)}"
                    bands_keys.append(key)
                    bands_records.append(np.asarray(bands, dtype=np.float32))
                    provenance[key] = metadata
    return slope_rows, bands_keys, bands_records, provenance, failures


def collect_gap_evidence(
    contracts: Mapping[str, Any],
    slope_rows: Sequence[Mapping[str, Any]],
    *,
    lanes: Sequence[str],
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect all-band analytic and slope-oracle gap-alignment evidence."""

    import h5py

    spacing_parts: list[np.ndarray] = []
    for lane in lanes:
        contract = contracts[lane]
        with h5py.File(contract.path, "r") as handle:
            values = np.asarray(handle["Wh"][contract.selected_train], dtype=np.float64)
            spacing = np.diff(values, axis=-1)
            spacing_parts.append(spacing[spacing > 0.0])
    spacing_floor = float(np.quantile(np.concatenate(spacing_parts), 0.25))
    slope_by_key = {
        (row["lane"], row["split"], int(row["dataset_index"])): row for row in slope_rows
    }
    gap_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for lane in lanes:
        contract = contracts[lane]
        with h5py.File(contract.path, "r") as handle:
            for split, indices in (("train", contract.selected_train), ("validation", contract.validation)):
                for index in indices:
                    Wh = np.asarray(handle["Wh"][int(index)], dtype=np.float64)
                    epsilon = epsilon_from_raster(
                        np.asarray(handle[contract.raster_field][int(index)]), contract.solver
                    )
                    old_candidates = candidate_values(float(np.mean(np.sqrt(epsilon))))
                    new_candidates = candidate_values(
                        float(effective_refractive_index(epsilon, "TE"))
                    )
                    for candidate in CANDIDATES:
                        old_gap = nearest_gap(Wh, old_candidates[candidate], spacing_floor)
                        new_gap = nearest_gap(Wh, new_candidates[candidate], spacing_floor)
                        gap_rows.append(
                            {
                                "lane": lane,
                                "split": split,
                                "dataset_index": int(index),
                                "candidate": candidate,
                                "old_frequency": old_candidates[candidate],
                                "new_frequency": new_candidates[candidate],
                                **{f"old_{key}": value for key, value in old_gap.items()},
                                **{f"new_{key}": value for key, value in new_gap.items()},
                                "distance_improvement": old_gap["normalized_center_distance"]
                                - new_gap["normalized_center_distance"],
                            }
                        )
                    slope = slope_by_key.get((lane, split, int(index)))
                    if slope is None:
                        continue
                    oracle_indices = {
                        "gamma_g1": float(slope["oracle_index_mean"]),
                        "x": float(slope["x_index"]),
                        "m": float(slope["m_index"]),
                    }
                    for candidate in CANDIDATES:
                        oracle_frequency = candidate_values(oracle_indices[candidate])[candidate]
                        oracle_gap = nearest_gap(Wh, oracle_frequency, spacing_floor)
                        new_gap = nearest_gap(Wh, new_candidates[candidate], spacing_floor)
                        oracle_rows.append(
                            {
                                "lane": lane,
                                "split": split,
                                "dataset_index": int(index),
                                "candidate": candidate,
                                "analytic_pair": int(new_gap["lower_band_one_based"]),
                                "oracle_pair": int(oracle_gap["lower_band_one_based"]),
                                "pair_match": bool(
                                    new_gap["lower_band_one_based"]
                                    == oracle_gap["lower_band_one_based"]
                                ),
                                "oracle_status": oracle_gap["status"],
                                "oracle_normalized_center_distance": oracle_gap[
                                    "normalized_center_distance"
                                ],
                                "oracle_out_of_range": oracle_gap["out_of_frequency_range"],
                            }
                        )
    return spacing_floor, gap_rows, oracle_rows


__all__ = [
    "StructureSolver",
    "UniformSolver",
    "collect_gap_evidence",
    "collect_slope_evidence",
    "epsilon_from_raster",
    "run_uniform_sentinels",
]
