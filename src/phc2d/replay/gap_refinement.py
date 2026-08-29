"""EXP-PHY-006 local full-BZ band-edge refinement pilot.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.full_cell import FullCellStructureSpec
from phc2d.hpc import sha256_file
from phc2d.physics.bz_gap_refinement import (
    adjacent_gap_records,
    compare_gap_records,
    propose_gap_extrema_refinement_points,
)
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_json,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.solvers.full_cell import FullCellMpbBandSolverConfig


LANES = ("binary", "graded")
BandSolver = Callable[..., tuple[np.ndarray, dict[str, Any]]]


@dataclass(frozen=True)
class RefinementSample:
    lane: str
    dataset_index: int
    sample_id: str
    geometry_hash: str
    structure: FullCellStructureSpec
    k_points: np.ndarray
    bands: np.ndarray


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    resolved = {key: dict(raw.get(key) or {}) for key in (
        "experiment", "datasets", "sampling", "mpb", "refinement", "gates", "execution", "output"
    )}
    if resolved["experiment"].get("experiment_id") != "EXP-PHY-006":
        raise ValueError("gap refinement requires experiment_id=EXP-PHY-006")
    if set(resolved["datasets"]) != set(LANES):
        raise ValueError("datasets must contain exactly binary and graded")
    if str(resolved["mpb"].get("polarization", "")).upper() != "TE":
        raise ValueError("EXP-PHY-006 pilot is frozen to TE")
    if int(resolved["refinement"].get("lower_band_min_one_based", 0)) != 1:
        raise ValueError("EXP-PHY-006 must include the B1-B2 gap")
    levels = int(resolved["refinement"].get("levels", 0))
    if levels < 1 or float(resolved["refinement"].get("initial_spacing", 0.0)) <= 0.0:
        raise ValueError("refinement requires positive levels and initial_spacing")
    mode = str(resolved["execution"].get("mode", "preflight"))
    if mode not in {"preflight", "pilot_real_mpb"}:
        raise ValueError("execution.mode must be preflight or pilot_real_mpb")
    if mode == "pilot_real_mpb" and resolved["execution"].get("formal_solver_authorized") is not True:
        raise ValueError("pilot real-MPB execution requires formal_solver_authorized=true")
    for lane in LANES:
        dataset = dict(resolved["datasets"][lane])
        samples = list(dataset.get("samples") or [])
        if not samples:
            raise ValueError(f"{lane} requires explicit validation samples")
        for sample in samples:
            if int(sample.get("dataset_index", -1)) < 0 or not sample.get("geometry_hash"):
                raise ValueError(f"{lane} sample identity is incomplete")
        resolved["datasets"][lane] = dataset
    return resolved


def _load_samples(resolved: Mapping[str, Any]) -> tuple[list[RefinementSample], list[dict[str, Any]]]:
    import h5py

    samples: list[RefinementSample] = []
    manifest: list[dict[str, Any]] = []
    for lane in LANES:
        dataset = dict(resolved["datasets"][lane])
        path = Path(str(dataset["path"]))
        actual_sha = sha256_file(path)
        if actual_sha != str(dataset.get("expected_dataset_sha256", "")):
            raise ValueError(f"{lane} dataset SHA-256 mismatch")
        with h5py.File(path, "r") as handle:
            required = {"Wh", "k_points", "idx_val", "idx_test", "sample_id", "geometry_hash", "structure_json"}
            missing = sorted(required.difference(handle.keys()))
            if missing:
                raise KeyError(f"{lane} dataset missing fields: {missing}")
            validation = set(int(value) for value in np.asarray(handle["idx_val"], dtype=np.int64))
            test = set(int(value) for value in np.asarray(handle["idx_test"], dtype=np.int64))
            k_points = np.asarray(handle["k_points"], dtype=np.float64)
            for identity in dataset["samples"]:
                index = int(identity["dataset_index"])
                if index not in validation or index in test:
                    raise ValueError(f"{lane} index {index} is not validation-only")
                geometry_hash = _decode(handle["geometry_hash"][index])
                if geometry_hash != str(identity["geometry_hash"]):
                    raise ValueError(f"{lane} geometry hash mismatch at index {index}")
                structure = FullCellStructureSpec.from_mapping(
                    json.loads(_decode(handle["structure_json"][index]))
                )
                if structure.geometry_hash != geometry_hash:
                    raise ValueError(f"{lane} structure payload hash mismatch at index {index}")
                sample_id = _decode(handle["sample_id"][index])
                samples.append(
                    RefinementSample(
                        lane=lane,
                        dataset_index=index,
                        sample_id=sample_id,
                        geometry_hash=geometry_hash,
                        structure=structure,
                        k_points=k_points.copy(),
                        bands=np.asarray(handle["Wh"][index], dtype=np.float64),
                    )
                )
                manifest.append(
                    {
                        "lane": lane,
                        "split": "validation",
                        "dataset_index": index,
                        "sample_id": sample_id,
                        "geometry_hash": geometry_hash,
                        "dataset_sha256": actual_sha,
                    }
                )
    return samples, manifest


def _solver_config(mpb: Mapping[str, Any], *, resolution: int, band_count: int) -> FullCellMpbBandSolverConfig:
    return FullCellMpbBandSolverConfig(
        num_bands=int(band_count),
        resolution=int(resolution),
        polarization=str(mpb["polarization"]),
        epsilon_air=float(mpb["epsilon_air"]),
        epsilon_material=float(mpb["epsilon_material"]),
        eigensolver_tolerance=float(mpb["eigensolver_tolerance"]),
        band_guard_count=int(mpb["band_guard_count"]),
        band_order_policy=str(mpb["band_order_policy"]),
        mesh_size=int(mpb["mesh_size"]),
    )


def _refine_one(
    sample: RefinementSample,
    *,
    resolved: Mapping[str, Any],
    solver: BandSolver,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    refinement = dict(resolved["refinement"])
    mpb = dict(resolved["mpb"])
    lower_min = int(refinement["lower_band_min_one_based"])
    lower_max_raw = refinement.get("lower_band_max_one_based")
    lower_max = int(lower_max_raw) if lower_max_raw is not None else None
    points = sample.k_points.copy()
    bands = sample.bands.copy()
    coarse = adjacent_gap_records(bands, lower_band_min_one_based=lower_min, lower_band_max_one_based=lower_max)
    previous = coarse
    history: list[dict[str, Any]] = []
    last_comparison: list[dict[str, Any]] = []
    primary_config = _solver_config(mpb, resolution=int(mpb["resolution"]), band_count=bands.shape[1])
    for level in range(1, int(refinement["levels"]) + 1):
        step = float(refinement["initial_spacing"]) / (2.0**level)
        proposed = propose_gap_extrema_refinement_points(
            points,
            bands,
            step=step,
            lower_band_min_one_based=lower_min,
            lower_band_max_one_based=lower_max,
        )
        if len(proposed):
            new_bands, _ = solver(sample.structure, k_points=proposed, config=primary_config)
            points = np.concatenate((points, proposed), axis=0)
            bands = np.concatenate((bands, np.asarray(new_bands, dtype=np.float64)), axis=0)
        current = adjacent_gap_records(
            bands,
            lower_band_min_one_based=lower_min,
            lower_band_max_one_based=lower_max,
        )
        last_comparison = compare_gap_records(previous, current)
        history.append(
            {
                "lane": sample.lane,
                "dataset_index": sample.dataset_index,
                "stage": f"refinement_{level}",
                "new_k_points": int(len(proposed)),
                "total_k_points": int(len(points)),
                "step": step,
                "max_gap_width_change": max(abs(float(row["gap_width_change"])) for row in last_comparison),
                "max_center_change": max(float(row["center_absolute_change"]) for row in last_comparison),
            }
        )
        previous = current
    refined = previous
    edge_indices = sorted(
        {
            int(row[key])
            for row in refined
            for key in ("lower_edge_k_index", "upper_edge_k_index")
        }
    )
    edge_points = points[edge_indices]
    verification_config = _solver_config(
        mpb,
        resolution=int(mpb["verification_resolution"]),
        band_count=bands.shape[1],
    )
    verification_bands, verification_meta = solver(
        sample.structure,
        k_points=edge_points,
        config=verification_config,
    )
    verification_bands = np.asarray(verification_bands, dtype=np.float64)
    edge_position = {index: position for position, index in enumerate(edge_indices)}
    coarse_comparison = compare_gap_records(coarse, refined)
    last_by_band = {int(row["lower_band_one_based"]): row for row in last_comparison}
    output_rows: list[dict[str, Any]] = []
    gates = dict(resolved["gates"])
    for comparison, final in zip(coarse_comparison, refined, strict=True):
        lower_band = int(final["lower_band_one_based"])
        lower_index = int(final["lower_edge_k_index"])
        upper_index = int(final["upper_edge_k_index"])
        lower_resolution_change = abs(
            float(bands[lower_index, lower_band - 1])
            - float(verification_bands[edge_position[lower_index], lower_band - 1])
        )
        upper_resolution_change = abs(
            float(bands[upper_index, lower_band])
            - float(verification_bands[edge_position[upper_index], lower_band])
        )
        last = last_by_band[lower_band]
        uncertainty = (
            abs(float(last["gap_width_change"]))
            + lower_resolution_change
            + upper_resolution_change
        )
        width = float(final["signed_width"])
        if width - uncertainty > 0.0:
            local_status = "locally_certified_complete"
        elif width + uncertainty < 0.0:
            local_status = "locally_certified_no_gap"
        else:
            local_status = "numerically_ambiguous"
        output_rows.append(
            {
                "lane": sample.lane,
                "dataset_index": sample.dataset_index,
                "sample_id": sample.sample_id,
                **comparison,
                "last_level_gap_width_change": float(last["gap_width_change"]),
                "last_level_center_change": float(last["center_absolute_change"]),
                "lower_resolution_change": lower_resolution_change,
                "upper_resolution_change": upper_resolution_change,
                "estimated_local_uncertainty": uncertainty,
                "local_certification_status": local_status,
                "last_refinement_stable": abs(float(last["gap_width_change"]))
                <= float(gates["last_refinement_gap_change_max"]),
                "resolution_stable": max(lower_resolution_change, upper_resolution_change)
                <= float(gates["resolution_edge_change_max"]),
            }
        )
    arrays = {"k_points": points, "bands": bands, "verification_k_points": edge_points, "verification_bands": verification_bands}
    provenance = {
        "primary_resolution": int(mpb["resolution"]),
        "verification_resolution": int(mpb["verification_resolution"]),
        "verification_solver": verification_meta,
    }
    return output_rows, arrays, history, provenance


def run_gap_refinement_pilot(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    preflight: bool = False,
    solver: BandSolver | None = None,
) -> Path:
    """Run EXP-PHY-006 validation-only preflight or local-refinement pilot."""

    config_file = Path(config_path)
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("EXP-PHY-006 config root must be a mapping")
    resolved = _validate_config(raw)
    mode = "preflight" if preflight else str(resolved["execution"]["mode"])
    out = make_run_dir("EXP-PHY-006", runs_dir=resolved["output"].get("runs_dir", "runs"), run_dir=run_dir)
    logger = setup_logging(log_file=out / RUN_LOG_NAME, run_meta={"script": "phc2d replay gap-refinement", "execution_mode": mode})
    started = time.perf_counter()
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)
    samples, manifest = _load_samples(resolved)
    _write_jsonl(out / "sample_manifest.jsonl", manifest)
    base_meta = {
        "script": "phc2d replay gap-refinement",
        "config_path": str(config_file),
        "execution_mode": mode,
        "test_targets_loaded": False,
        "training_started": False,
        "n40_used": False,
        "sample_count": len(samples),
        "sample_manifest_sha256": sha256_file(out / "sample_manifest.jsonl"),
    }
    if mode == "preflight":
        _write_csv(out / "history.csv", [{"stage": "preflight", "status": "pass"}])
        _write_jsonl(out / "solver_failures.jsonl", [])
        _write_csv(out / "gap_refinement_samples.csv", [])
        np.savez_compressed(out / "refined_bands.npz", status=np.asarray(["not_run"]))
        write_json(out / "refinement_metrics.json", {"status": "not_run"})
        write_metrics(out, {"status": "preflight_pass", "sample_count": len(samples)})
        write_results(out, {"status": "preflight_pass", "formal_physical_validation_completed": False})
        write_run_meta(out, base_meta)
        return out
    if solver is None:
        from phc2d.solvers.full_cell import solve_bands_for_full_cell_mpb

        solver = solve_bands_for_full_cell_mpb
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for sample in samples:
        logger.info("refining %s index=%d", sample.lane, sample.dataset_index)
        try:
            sample_rows, sample_arrays, sample_history, sample_provenance = _refine_one(
                sample,
                resolved=resolved,
                solver=solver,
            )
        except Exception as exc:
            failures.append(
                {
                    "lane": sample.lane,
                    "dataset_index": sample.dataset_index,
                    "sample_id": sample.sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        key = f"{sample.lane}_{sample.dataset_index}"
        rows.extend(sample_rows)
        history.extend(sample_history)
        arrays.update({f"{key}_{name}": value for name, value in sample_arrays.items()})
        provenance[key] = sample_provenance
    _write_jsonl(out / "solver_failures.jsonl", failures)
    _write_csv(out / "gap_refinement_samples.csv", rows)
    _write_csv(out / "history.csv", history or [{"stage": "solver", "status": "blocked"}])
    np.savez_compressed(out / "refined_bands.npz", **arrays)
    elapsed = time.perf_counter() - started
    status_changes = sum(bool(row["status_changed"]) for row in rows)
    large_gap_changes = sum(
        abs(float(row["gap_width_change"])) > float(resolved["gates"]["coarse_gap_change_material"])
        for row in rows
    )
    stable = bool(rows) and all(
        bool(row["last_refinement_stable"]) and bool(row["resolution_stable"])
        for row in rows
    )
    if failures:
        status = "blocked"
    elif status_changes or large_gap_changes:
        status = "coarse_grid_risk_detected"
    elif stable:
        status = "pilot_local_refinement_stable"
    else:
        status = "pilot_inconclusive"
    metrics = {
        "status": status,
        "sample_count": len(samples),
        "completed_sample_count": len(samples) - len(failures),
        "gap_pair_row_count": len(rows),
        "status_change_count": status_changes,
        "material_gap_change_count": large_gap_changes,
        "max_coarse_gap_overestimate": max((float(row["coarse_gap_overestimate"]) for row in rows), default=None),
        "max_center_absolute_change": max((float(row["center_absolute_change"]) for row in rows), default=None),
        "last_refinement_all_stable": stable,
        "wall_time_total_s": elapsed,
    }
    write_json(out / "refinement_metrics.json", metrics)
    write_metrics(out, metrics)
    write_results(
        out,
        {
            "status": status,
            "formal_physical_validation_completed": False,
            "pilot_real_mpb_completed": not failures,
            "evidence_boundary": "validation_only_local_extrema_refinement_not_global_BZ_certification",
            "training_started": False,
            "test_targets_loaded": False,
            "n40_used": False,
            "next_gate": "human_review_before_64_structure_global_adaptive_BZ_study",
        },
    )
    write_run_meta(out, {**base_meta, "solver_provenance": provenance})
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete EXP-PHY-006 artifacts: {missing}")
    return out


__all__ = ["BandSolver", "run_gap_refinement_pilot"]
