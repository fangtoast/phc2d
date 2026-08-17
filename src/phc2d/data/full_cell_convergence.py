"""Resolution-convergence gate for full-cell real-MPB datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from phc2d.data.full_cell import FullCellDatasetBundle, FullCellStructureSpec, load_full_cell_hdf5
from phc2d.data.full_cell_generation import (
    _file_hash,
    _jsonable,
    _relative,
    _repo_path,
    _solver_config,
    _validate_execution_boundary,
    write_combined_full_cell_manifest,
)
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
from phc2d.solvers import MPBParallelConfig, MPBWorkerJob, run_mpb_worker_jobs, solve_bands_for_full_cell_mpb


FullCellSolverFn = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def _stratum(structure: FullCellStructureSpec) -> str:
    if structure.generator_family == "graded_rbf_fourier":
        core_count = structure.metadata.get("core_count")
        if core_count is None:
            core_count = len(list(structure.parameters.get("blobs") or []))
        return f"graded_core_{int(core_count)}"
    if structure.generator_family == "binary_polar_holes":
        hole_count = structure.metadata.get("hole_count")
        if hole_count is None:
            hole_count = structure.parameters.get("hole_count")
        if hole_count is None:
            hole_count = len(list(structure.parameters.get("holes") or []))
        return f"binary_polar_holes_{int(hole_count)}"
    return "binary_spectral_threshold"


def select_convergence_indices(structures: Sequence[FullCellStructureSpec], count: int) -> np.ndarray:
    """Select a deterministic round-robin sample across generator strata."""

    requested = int(count)
    if requested <= 0 or requested > len(structures):
        raise ValueError("convergence sample count must be positive and no larger than the dataset")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, structure in enumerate(structures):
        groups[_stratum(structure)].append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: (structures[index].geometry_hash, structures[index].sample_id))
    selected: list[int] = []
    depth = 0
    keys = sorted(groups)
    while len(selected) < requested:
        progressed = False
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                progressed = True
                if len(selected) == requested:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected) != requested:
        raise RuntimeError("unable to select the requested convergence samples")
    return np.asarray(selected, dtype=np.int64)


def select_convergence_indices_by_stratum(
    structures: Sequence[FullCellStructureSpec],
    stratum_counts: Mapping[str, int],
) -> np.ndarray:
    """Select exact deterministic counts from named generator strata."""

    requested = {str(key): int(value) for key, value in stratum_counts.items()}
    if not requested or any(value <= 0 for value in requested.values()):
        raise ValueError("stratum_counts must contain positive counts")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, structure in enumerate(structures):
        groups[_stratum(structure)].append(index)
    selected: list[int] = []
    for stratum, count in requested.items():
        candidates = sorted(
            groups.get(stratum, []),
            key=lambda index: (structures[index].geometry_hash, structures[index].sample_id),
        )
        if len(candidates) < count:
            raise ValueError(
                f"stratum {stratum!r} provides {len(candidates)} structures, fewer than requested {count}"
            )
        positions = (
            np.asarray([0], dtype=np.int64)
            if count == 1
            else np.linspace(0, len(candidates) - 1, num=count, dtype=np.int64)
        )
        selected.extend(candidates[int(position)] for position in positions)
    if len(set(selected)) != len(selected):
        raise RuntimeError("stratified convergence selection contains duplicate indices")
    return np.asarray(selected, dtype=np.int64)


def representative_k_indices(k_grid_index: np.ndarray) -> np.ndarray:
    """Return center, axis edges, and corners from an odd square full-BZ grid."""

    grid = np.asarray(k_grid_index, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("k_grid_index must have shape [n_k_points, 2]")
    half = int(np.max(np.abs(grid)))
    wanted = {
        (-half, -half), (-half, 0), (-half, half),
        (0, -half), (0, 0), (0, half),
        (half, -half), (half, 0), (half, half),
    }
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(grid)}
    missing = sorted(wanted - set(lookup))
    if missing:
        raise ValueError(f"full-BZ grid is missing representative indices: {missing}")
    return np.asarray([lookup[key] for key in sorted(wanted)], dtype=np.int64)


def compute_convergence_summary(
    bands_by_resolution: Mapping[int, np.ndarray],
    *,
    median_64_96_max: float,
    median_96_128_max: float,
    p95_max: float,
) -> dict[str, Any]:
    """Compute bands-2+ relative-error gates for one lane or combined pool."""

    required = (64, 96, 128)
    arrays = {resolution: np.asarray(bands_by_resolution[resolution], dtype=np.float64) for resolution in required}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("resolution-convergence arrays must have identical shapes")
    if arrays[64].ndim != 3 or arrays[64].shape[-1] < 2:
        raise ValueError("convergence arrays must have shape [samples, k_points, bands>=2]")

    def relative(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
        return np.abs(coarse[..., 1:] - fine[..., 1:]) / np.maximum(np.abs(fine[..., 1:]), 1.0e-8)

    error_64_96 = relative(arrays[64], arrays[96])
    error_96_128 = relative(arrays[96], arrays[128])
    metrics = {
        "median_mre_64_96": float(np.median(error_64_96)),
        "median_mre_96_128": float(np.median(error_96_128)),
        "p95_mre_64_96": float(np.quantile(error_64_96, 0.95)),
        "p95_mre_96_128": float(np.quantile(error_96_128, 0.95)),
        "bands_in_gate": list(range(2, arrays[64].shape[-1] + 1)),
        "element_count_per_transition": int(error_64_96.size),
    }
    failures = []
    for name, threshold in (
        ("median_mre_64_96", float(median_64_96_max)),
        ("median_mre_96_128", float(median_96_128_max)),
        ("p95_mre_64_96", float(p95_max)),
        ("p95_mre_96_128", float(p95_max)),
    ):
        if not np.isfinite(metrics[name]) or float(metrics[name]) > threshold:
            failures.append(f"{name}={metrics[name]} exceeds {threshold}")
    return {**metrics, "passed": not failures, "failures": failures}


def compute_reference_convergence_summary(
    bands_by_resolution: Mapping[int, np.ndarray],
    *,
    reference_resolution: int,
    median_max: float,
    combined_p95_max: float,
    lane_p95_max: float,
    scope: str,
) -> dict[str, Any]:
    """Compare every lower resolution with one fixed numerical reference."""

    reference = int(reference_resolution)
    resolutions = sorted(int(value) for value in bands_by_resolution)
    if reference not in resolutions or resolutions[-1] != reference:
        raise ValueError("reference_resolution must be the highest configured resolution")
    arrays = {
        resolution: np.asarray(bands_by_resolution[resolution], dtype=np.float64)
        for resolution in resolutions
    }
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("resolution-convergence arrays must have identical shapes")
    if arrays[reference].ndim != 3 or arrays[reference].shape[-1] < 2:
        raise ValueError("convergence arrays must have shape [samples, k_points, bands>=2]")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("resolution-convergence arrays must be finite")

    comparisons: dict[str, Any] = {}
    failures: list[str] = []
    for resolution in resolutions[:-1]:
        error = np.abs(arrays[resolution][..., 1:] - arrays[reference][..., 1:]) / np.maximum(
            np.abs(arrays[reference][..., 1:]), 1.0e-8
        )
        metrics = {
            "median_mre": float(np.median(error)),
            "p90_mre": float(np.quantile(error, 0.90)),
            "p95_mre": float(np.quantile(error, 0.95)),
            "p99_mre": float(np.quantile(error, 0.99)),
            "max_mre": float(np.max(error)),
            "element_count": int(error.size),
        }
        threshold = float(combined_p95_max if scope == "combined" else lane_p95_max)
        passed = metrics["p95_mre"] <= threshold
        if scope == "combined":
            passed = passed and metrics["median_mre"] <= float(median_max)
        metrics["passed_candidate_gate"] = bool(passed)
        metrics["candidate_gate"] = {
            "median_mre_max": float(median_max) if scope == "combined" else None,
            "p95_mre_max": threshold,
        }
        comparisons[str(resolution)] = metrics
        if not passed:
            failures.append(
                f"r{resolution}->r{reference} median={metrics['median_mre']} p95={metrics['p95_mre']}"
            )
    return {
        "reference_resolution": reference,
        "bands_in_scope": list(range(2, arrays[reference].shape[-1] + 1)),
        "comparisons": comparisons,
        "all_lower_resolutions_pass_candidate_gate": not failures,
        "failures": failures,
    }


def _validate_config(cfg: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    experiment = dict(cfg.get("experiment") or {})
    if experiment.get("stage") != "resolution_convergence":
        problems.append("experiment.stage must be resolution_convergence")
    if not experiment.get("experiment_id"):
        problems.append("experiment.experiment_id is required")
    datasets = dict(cfg.get("datasets") or {})
    for lane in ("graded_medium", "binary_air_holes"):
        if lane not in datasets:
            problems.append(f"datasets.{lane} is required")
        elif not _repo_path(str(datasets[lane])).is_file():
            problems.append(f"datasets.{lane} does not exist: {datasets[lane]}")
    resolutions = [int(value) for value in cfg.get("resolutions", [])]
    if len(resolutions) < 3 or resolutions != sorted(set(resolutions)) or min(resolutions, default=0) <= 0:
        problems.append("resolutions must contain at least three unique positive ascending values")
    samples_per_lane = int(dict(cfg.get("selection") or {}).get("samples_per_lane", 0))
    if samples_per_lane <= 0:
        problems.append("selection.samples_per_lane must be positive")
    stratum_counts = dict(dict(cfg.get("selection") or {}).get("stratum_counts") or {})
    for lane, counts in stratum_counts.items():
        if lane not in {"graded_medium", "binary_air_holes"}:
            problems.append(f"selection.stratum_counts has unsupported lane: {lane}")
            continue
        parsed_counts = {str(key): int(value) for key, value in dict(counts or {}).items()}
        if not parsed_counts or any(value <= 0 for value in parsed_counts.values()):
            problems.append(f"selection.stratum_counts.{lane} must contain positive counts")
        elif sum(parsed_counts.values()) != samples_per_lane:
            problems.append(f"selection.stratum_counts.{lane} must sum to selection.samples_per_lane")
    solver = dict(cfg.get("solver") or {})
    required_solver = {
        "backend", "environment", "use_proxy", "epsilon_air", "epsilon_material", "resolution",
        "mesh_size", "polarization", "num_bands", "band_guard_count",
        "band_order_policy", "eigensolver_tolerance",
    }
    missing = sorted(required_solver - set(solver))
    if missing:
        problems.append("missing solver fields: " + ", ".join(missing))
    if str(solver.get("backend", "")).lower() != "mpb" or bool(solver.get("use_proxy", False)):
        problems.append("solver must be real MPB with use_proxy=false")
    if str(solver.get("environment", "")) not in {"wsl_meep_env", "native_linux_conda_meep_env"}:
        problems.append("solver.environment must be wsl_meep_env or native_linux_conda_meep_env")
    if str(solver.get("polarization", "")).upper() != "TE":
        problems.append("resolution convergence is frozen to TE")
    for key, expected in {"mesh_size": 3, "num_bands": 10, "band_guard_count": 4}.items():
        if int(solver.get(key, -1)) != expected:
            problems.append(f"solver.{key} must be {expected}")
    if resolutions and int(solver.get("resolution", -1)) != resolutions[0]:
        problems.append("solver.resolution must equal the first configured resolution")
    selection = dict(cfg.get("selection") or {})
    if str(selection.get("k_policy", "")) not in {"full_bz_all", "full_bz_center_axis_edges_and_corners"}:
        problems.append("selection.k_policy must be full_bz_all or full_bz_center_axis_edges_and_corners")
    if str(solver.get("band_order_policy", "")) != "frequency_ascending":
        problems.append("solver.band_order_policy must be frequency_ascending")
    if not np.isclose(float(solver.get("eigensolver_tolerance", np.nan)), 1.0e-7):
        problems.append("solver.eigensolver_tolerance must be 1e-7")
    return problems


def _write_reference_figure(
    path: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    solve_time_by_resolution: Mapping[str, float],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for lane, summary in summaries.items():
        resolutions = [int(value) for value in summary["comparisons"]]
        axes[0].plot(
            resolutions,
            [100.0 * float(summary["comparisons"][str(value)]["p95_mre"]) for value in resolutions],
            marker="o",
            label=lane,
        )
    axes[0].set_xlabel("MPB resolution")
    axes[0].set_ylabel("P95 relative error (%)")
    axes[0].set_title("Bands 2-10 vs reference")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    timing_resolutions = [int(value) for value in solve_time_by_resolution]
    axes[1].plot(
        timing_resolutions,
        [float(solve_time_by_resolution[str(value)]) for value in timing_resolutions],
        marker="o",
    )
    axes[1].set_xlabel("MPB resolution")
    axes[1].set_ylabel("summed worker solve time (s)")
    axes[1].set_title("Stage-A compute cost")
    axes[1].grid(alpha=0.25)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_full_cell_resolution_convergence(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    skip_environment_check: bool = False,
    solver_fn: FullCellSolverFn = solve_bands_for_full_cell_mpb,
) -> Path:
    """Re-solve selected structures at 96/128 and gate them against stored 64."""

    config_path = _repo_path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = dict(cfg.get("experiment") or {})
    out = make_run_dir(
        str(experiment.get("experiment_id", "full_cell_resolution_convergence")),
        runs_dir=_repo_path(dict(cfg.get("output") or {}).get("runs_dir", "runs")),
        run_dir=_repo_path(run_dir) if run_dir else None,
    )
    logger = setup_logging(log_file=out / "run.log", run_meta={"script": "phc2d.data.full_cell_convergence"})
    ensure_figures_dir(out)
    started = time.perf_counter()
    write_resolved_config(out, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    problems = _validate_config(cfg)
    solver_cfg_payload = dict(cfg.get("solver") or {})
    execution_profile: dict[str, Any] = {}
    if not skip_environment_check:
        boundary_problems, execution_profile = _validate_execution_boundary(solver_cfg_payload)
        problems.extend(boundary_problems)
    if problems:
        reason = "; ".join(problems)
        _write_history(out / "history.csv", [{"stage": "contract_check", "status": "blocked", "reason": reason}])
        write_metrics(out, {"status": "blocked", "wall_time_total_s": float(time.perf_counter() - started)})
        write_results(out, {"status": "blocked", "reason": reason})
        write_run_meta(out, {"status": "blocked", "validation_boundary": "wsl_meep_env_real_mpb"})
        return out

    resolutions = [int(value) for value in cfg["resolutions"]]
    reference_resolution = int(dict(cfg.get("gates") or {}).get("reference_resolution", resolutions[-1]))
    datasets: dict[str, FullCellDatasetBundle] = {
        lane: load_full_cell_hdf5(_repo_path(path))
        for lane, path in dict(cfg["datasets"]).items()
    }
    for lane, bundle in datasets.items():
        if str(bundle.metadata.get("lane", "")) != lane:
            raise ValueError(f"dataset lane metadata mismatch for {lane}")
        dataset_solver = dict(bundle.metadata.get("solver") or {})
        if int(dataset_solver.get("resolution", -1)) != 64:
            raise ValueError(f"dataset {lane} is not a resolution-64 source")
        for key in ("epsilon_air", "epsilon_material"):
            if not np.isclose(float(dataset_solver.get(key, np.nan)), float(solver_cfg_payload.get(key, np.nan))):
                raise ValueError(f"dataset {lane} solver.{key} does not match convergence config")
        if bundle.Wh.ndim != 3 or bundle.Wh.shape[1:] != (81, 10):
            raise ValueError(f"dataset {lane} Wh must have shape [N,81,10]")
        if representative_k_indices(bundle.k_grid_index).shape != (9,):
            raise ValueError(f"dataset {lane} does not expose the required full-BZ grid")
        if not np.all(np.isfinite(bundle.Wh)) or np.any(np.diff(bundle.Wh, axis=-1) < -1.0e-7):
            raise ValueError(f"dataset {lane} Wh failed finite/ascending checks")
    samples_per_lane = int(dict(cfg["selection"])["samples_per_lane"])
    configured_strata = dict(dict(cfg["selection"]).get("stratum_counts") or {})
    selections = {
        lane: (
            select_convergence_indices_by_stratum(bundle.structures, dict(configured_strata[lane]))
            if lane in configured_strata
            else select_convergence_indices(bundle.structures, samples_per_lane)
        )
        for lane, bundle in datasets.items()
    }
    k_policy = str(dict(cfg["selection"])["k_policy"])
    k_indices_by_lane = {
        lane: (
            np.arange(bundle.k_points.shape[0], dtype=np.int64)
            if k_policy == "full_bz_all"
            else representative_k_indices(bundle.k_grid_index)
        )
        for lane, bundle in datasets.items()
    }
    jobs: list[MPBWorkerJob] = []
    job_metadata: dict[int, dict[str, Any]] = {}
    for lane in sorted(datasets):
        bundle = datasets[lane]
        k_indices = k_indices_by_lane[lane]
        for dataset_index in selections[lane]:
            structure = bundle.structures[int(dataset_index)]
            for resolution in resolutions:
                job_index = len(jobs)
                job_metadata[job_index] = {
                    "lane": lane,
                    "dataset_index": int(dataset_index),
                    "sample_id": structure.sample_id,
                    "resolution": resolution,
                }
                jobs.append(
                    MPBWorkerJob(
                        sample_id=f"{lane}_r{resolution}_{structure.sample_id}",
                        sample_index=job_index,
                        split="convergence",
                        rho=np.asarray(structure.parameter_vector, dtype=np.float32),
                        solver_input=structure,
                        k_points=bundle.k_points[k_indices],
                        solver_config=_solver_config(solver_cfg_payload, resolution_override=resolution),
                        rho_hash=hashlib.sha256(np.asarray(structure.parameter_vector, dtype=np.float64).tobytes()).hexdigest(),
                        geometry_hash=structure.geometry_hash,
                    )
                )
    selection_payload = {
        lane: {
            "dataset_path": _relative(_repo_path(dict(cfg["datasets"])[lane])),
            "dataset_sha256": _file_hash(_repo_path(dict(cfg["datasets"])[lane])),
            "sample_indices": selections[lane].tolist(),
            "sample_ids": [datasets[lane].structures[int(index)].sample_id for index in selections[lane]],
            "strata": [_stratum(datasets[lane].structures[int(index)]) for index in selections[lane]],
            "k_indices": k_indices_by_lane[lane].tolist(),
            "k_grid_indices": datasets[lane].k_grid_index[k_indices_by_lane[lane]].tolist(),
        }
        for lane in sorted(datasets)
    }
    write_json(out / "convergence_selection.json", selection_payload)
    with (out / "sample_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["lane", "dataset_sha256", "dataset_index", "sample_id", "geometry_hash", "stratum"],
        )
        writer.writeheader()
        for lane in sorted(datasets):
            for dataset_index in selections[lane]:
                structure = datasets[lane].structures[int(dataset_index)]
                writer.writerow(
                    {
                        "lane": lane,
                        "dataset_sha256": selection_payload[lane]["dataset_sha256"],
                        "dataset_index": int(dataset_index),
                        "sample_id": structure.sample_id,
                        "geometry_hash": structure.geometry_hash,
                        "stratum": _stratum(structure),
                    }
                )
    with (out / "job_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_index", "lane", "dataset_index", "sample_id", "resolution", "k_point_count"],
        )
        writer.writeheader()
        for job_index, metadata in sorted(job_metadata.items()):
            writer.writerow({"job_index": job_index, **metadata, "k_point_count": len(jobs[job_index].k_points)})
    parallel = MPBParallelConfig.from_mapping(dict(cfg.get("workers") or {}))
    results, worker_summary = run_mpb_worker_jobs(jobs, config=parallel, shard_dir=out / "worker_shards", solver_fn=solver_fn)
    failures = [
        {**job_metadata[result.sample_index], "status": result.status, "reason": result.error}
        for result in results
        if result.status != "computed" or result.bands is None
    ]
    (out / "failed_case_ledger.jsonl").write_text(
        "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    with (out / "timing.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_index", "lane", "sample_id", "resolution", "status", "solve_time_s", "resumed", "worker_id", "pid"],
        )
        writer.writeheader()
        for result in results:
            metadata = job_metadata[result.sample_index]
            writer.writerow(
                {
                    "job_index": result.sample_index,
                    "lane": metadata["lane"],
                    "sample_id": metadata["sample_id"],
                    "resolution": metadata["resolution"],
                    "status": result.status,
                    "solve_time_s": float(result.solve_time_s),
                    "resumed": bool(result.resumed),
                    "worker_id": result.worker_id,
                    "pid": result.pid,
                }
            )
    if failures or len(results) != len(jobs):
        reason = failures[0]["reason"] if failures else "worker result count mismatch"
        _write_history(out / "history.csv", [{"stage": "resolution_convergence", "status": "solver_execution_failed", "reason": reason}])
        write_metrics(out, {**worker_summary, "status": "solver_execution_failed", "wall_time_total_s": float(time.perf_counter() - started)})
        write_results(out, {"status": "solver_execution_failed", "reason": reason})
        write_run_meta(out, {"status": "solver_execution_failed", "validation_boundary": "wsl_meep_env_real_mpb"})
        return out

    solved = {result.sample_index: np.asarray(result.bands, dtype=np.float32) for result in results}
    lane_arrays: dict[str, dict[int, np.ndarray]] = {}
    for lane in sorted(datasets):
        bundle = datasets[lane]
        selection = selections[lane]
        k_indices = k_indices_by_lane[lane]
        arrays: dict[int, np.ndarray] = {}
        for resolution in resolutions:
            rows = [
                solved[index]
                for index, metadata in sorted(job_metadata.items())
                if metadata["lane"] == lane and metadata["resolution"] == resolution
            ]
            arrays[resolution] = np.stack(rows, axis=0)
        lane_arrays[lane] = arrays
    gates = dict(cfg.get("gates") or {})
    gate_args = {
        "reference_resolution": reference_resolution,
        "median_max": float(gates.get("combined_median_mre_max", 0.001)),
        "combined_p95_max": float(gates.get("combined_p95_mre_max", 0.005)),
        "lane_p95_max": float(gates.get("lane_p95_mre_max", 0.0075)),
    }
    summaries = {
        lane: compute_reference_convergence_summary(arrays, scope=lane, **gate_args)
        for lane, arrays in lane_arrays.items()
    }
    combined_arrays = {
        resolution: np.concatenate([lane_arrays[lane][resolution] for lane in sorted(lane_arrays)], axis=0)
        for resolution in resolutions
    }
    summaries["combined"] = compute_reference_convergence_summary(combined_arrays, scope="combined", **gate_args)
    real_solver_execution = not skip_environment_check and solver_fn is solve_bands_for_full_cell_mpb
    acceptance_eligible = bool(execution_profile.get("formal_acceptance_eligible", False)) and real_solver_execution
    stage = str(experiment.get("phase", "legacy_gate"))
    candidate_resolutions = [
        resolution
        for resolution in resolutions[:-1]
        if all(
            bool(summary["comparisons"][str(resolution)]["passed_candidate_gate"])
            for summary in summaries.values()
        )
    ]
    recommendation_resolutions = [
        int(value) for value in gates.get("recommendation_resolutions", [64, 96])
    ]
    recommendation_candidates = [
        resolution for resolution in recommendation_resolutions if resolution in candidate_resolutions
    ]
    recommended_resolution = recommendation_candidates[0] if recommendation_candidates else None
    gate_passed = recommended_resolution is not None
    if stage == "stage_a_resource_pilot" and real_solver_execution:
        status = "stage_a_resource_pilot_complete_pending_review"
    elif stage == "stage_b_formal_confirmation" and real_solver_execution:
        status = "cluster_stage_b_complete_pending_wsl_acceptance" if gate_passed else "blocked"
    elif acceptance_eligible:
        status = "resolution_convergence_pass" if gate_passed else "blocked"
    else:
        status = "test_solver_smoke_only" if gate_passed else "blocked"
    np.savez_compressed(
        out / "convergence_bands.npz",
        **{
            f"{lane}_r{resolution}": array
            for lane, arrays in lane_arrays.items()
            for resolution, array in arrays.items()
        },
    )
    solve_time_by_resolution = {
        str(resolution): float(
            sum(
                result.solve_time_s
                for result in results
                if int(job_metadata[result.sample_index]["resolution"]) == resolution and not result.resumed
            )
        )
        for resolution in resolutions
    }
    resource_estimate = {
        "pilot_structure_count": int(samples_per_lane * len(datasets)),
        "formal_structure_count": 100,
        "estimated_formal_worker_cpu_hours": float(sum(solve_time_by_resolution.values()) / 3600.0 * 100.0 / (samples_per_lane * len(datasets))),
        "estimated_formal_wall_time_s_same_throughput": float(worker_summary["worker_wall_time_s"] * 100.0 / (samples_per_lane * len(datasets))),
        "solve_time_s_by_resolution": solve_time_by_resolution,
        "requires_human_acceptance": True,
    }
    write_json(
        out / "convergence_metrics.json",
        {
            "status": status,
            "gates": gate_args,
            "lanes": summaries,
            "candidate_resolutions": candidate_resolutions,
            "recommendation_resolutions": recommendation_resolutions,
            "provisional_recommended_resolution": recommended_resolution if stage == "stage_a_resource_pilot" else None,
            "recommended_resolution": recommended_resolution if stage == "stage_b_formal_confirmation" else None,
            "resource_estimate": resource_estimate,
        },
    )
    _write_reference_figure(out / "figures" / "resolution_convergence.png", summaries, solve_time_by_resolution)
    reason = "; ".join(f"{lane}: {failure}" for lane, summary in summaries.items() for failure in summary["failures"])
    _write_history(out / "history.csv", [{"stage": "resolution_convergence", "status": status, "reason": reason}])
    write_metrics(
        out,
        {
            **worker_summary,
            "status": status,
            "wall_time_total_s": float(time.perf_counter() - started),
            "mpb_generation_time_s": float(sum(result.solve_time_s for result in results if not result.resumed)),
            "acceptance_eligible": acceptance_eligible,
            "real_solver_execution": real_solver_execution,
            "summaries": summaries,
            "resource_estimate": resource_estimate,
        },
    )
    write_results(
        out,
        {
            "status": status,
            "gate_passed": gate_passed,
            "candidate_resolutions": candidate_resolutions,
            "recommendation_resolutions": recommendation_resolutions,
            "provisional_recommended_resolution": recommended_resolution if stage == "stage_a_resource_pilot" else None,
            "recommended_resolution": recommended_resolution if stage == "stage_b_formal_confirmation" else None,
            "next_gate": (
                "human_accept_stage_a_resource_report_and_authorize_stage_b"
                if stage == "stage_a_resource_pilot"
                else "wsl_cross_environment_acceptance"
                if stage == "stage_b_formal_confirmation" and status != "blocked"
                else "publish_combined_manifest"
                if status == "resolution_convergence_pass"
                else "inspect_resolution_failures"
            ),
        },
    )
    write_run_meta(
        out,
        {
            "status": status,
            "validation_boundary": str(execution_profile.get("validation_boundary", "unaccepted_test_only")),
            "config_path": _relative(config_path),
            "execution_profile": execution_profile,
            "phase": stage,
        },
    )
    logger.info("Full-cell resolution convergence completed: %s", status)
    return out


def await_full_cell_pilots_and_finalize(
    *,
    graded_run_dir: str | Path,
    binary_run_dir: str | Path,
    convergence_config: str | Path,
    combined_manifest: str | Path,
    poll_interval_s: float = 60.0,
    timeout_s: float = 8.0 * 60.0 * 60.0,
    convergence_runner: Callable[..., Path] = run_full_cell_resolution_convergence,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Path:
    """Wait for two fixed pilot runs, then run convergence and gated collection promotion."""

    runs = {
        "graded_medium": _repo_path(graded_run_dir),
        "binary_air_holes": _repo_path(binary_run_dir),
    }
    if float(poll_interval_s) < 0.0 or float(timeout_s) <= 0.0:
        raise ValueError("poll_interval_s must be non-negative and timeout_s must be positive")
    deadline = time.monotonic() + float(timeout_s)
    expected_status = "pilot_generated_pending_convergence"
    while True:
        statuses: dict[str, str] = {}
        for lane, run in runs.items():
            result_path = run / "results.json"
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                statuses[lane] = str(payload.get("status", ""))
                if statuses[lane] != expected_status:
                    raise RuntimeError(f"{lane} pilot stopped with status={statuses[lane]!r}: {result_path}")
        if len(statuses) == len(runs):
            break
        if time.monotonic() >= deadline:
            missing = sorted(set(runs) - set(statuses))
            raise TimeoutError(f"timed out waiting for full-cell pilots: {missing}")
        sleep_fn(float(poll_interval_s))

    for lane, run in runs.items():
        manifest_path = run / "dataset_manifest.yaml"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{lane} pilot manifest is missing: {manifest_path}")
    convergence_run = convergence_runner(_repo_path(convergence_config))
    evidence_path = Path(convergence_run) / "convergence_metrics.json"
    if not evidence_path.is_file():
        raise FileNotFoundError(f"convergence evidence is missing: {evidence_path}")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if str(evidence.get("status", "")) != "resolution_convergence_pass":
        raise RuntimeError(f"resolution convergence did not pass: {evidence_path}")
    collection_path = write_combined_full_cell_manifest(
        combined_manifest,
        graded_manifest=runs["graded_medium"] / "dataset_manifest.yaml",
        binary_manifest=runs["binary_air_holes"] / "dataset_manifest.yaml",
        convergence_evidence=evidence_path,
        status="pilot_pass",
    )
    write_json(
        Path(convergence_run) / "finalization.json",
        {
            "status": "pilot_pass",
            "graded_run_dir": _relative(runs["graded_medium"]),
            "binary_run_dir": _relative(runs["binary_air_holes"]),
            "combined_manifest": _relative(collection_path),
            "combined_manifest_sha256": _file_hash(collection_path),
        },
    )
    return Path(convergence_run)


__all__ = [
    "await_full_cell_pilots_and_finalize",
    "compute_convergence_summary",
    "compute_reference_convergence_summary",
    "representative_k_indices",
    "run_full_cell_resolution_convergence",
    "select_convergence_indices",
    "select_convergence_indices_by_stratum",
]
