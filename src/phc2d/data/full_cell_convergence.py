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
        return f"graded_core_{int(structure.metadata.get('core_count', 0))}"
    if structure.generator_family == "binary_polar_holes":
        return f"polar_holes_{int(structure.metadata.get('hole_count', 0))}"
    return "spectral_threshold"


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
    if resolutions != [64, 96, 128]:
        problems.append("resolutions must be exactly [64, 96, 128]")
    if int(dict(cfg.get("selection") or {}).get("samples_per_lane", 0)) <= 0:
        problems.append("selection.samples_per_lane must be positive")
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
    if str(solver.get("environment", "")) != "wsl_meep_env":
        problems.append("solver.environment must be wsl_meep_env")
    if str(solver.get("polarization", "")).upper() != "TE":
        problems.append("resolution convergence is frozen to TE")
    for key, expected in {"resolution": 64, "mesh_size": 3, "num_bands": 10, "band_guard_count": 4}.items():
        if int(solver.get(key, -1)) != expected:
            problems.append(f"solver.{key} must be {expected}")
    if str(solver.get("band_order_policy", "")) != "frequency_ascending":
        problems.append("solver.band_order_policy must be frequency_ascending")
    if not np.isclose(float(solver.get("eigensolver_tolerance", np.nan)), 1.0e-7):
        problems.append("solver.eigensolver_tolerance must be 1e-7")
    return problems


def _write_figure(path: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = list(summaries)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.18
    fields = ("median_mre_64_96", "median_mre_96_128", "p95_mre_64_96", "p95_mre_96_128")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for offset, field in enumerate(fields):
        axis.bar(x + (offset - 1.5) * width, [100.0 * float(summaries[label][field]) for label in labels], width, label=field)
    axis.set_xticks(x, labels)
    axis.set_ylabel("relative error (%)")
    axis.set_title("Full-cell MPB resolution convergence, bands 2+")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
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
    if not skip_environment_check:
        problems.extend(_validate_execution_boundary())
    if problems:
        reason = "; ".join(problems)
        _write_history(out / "history.csv", [{"stage": "contract_check", "status": "blocked", "reason": reason}])
        write_metrics(out, {"status": "blocked", "wall_time_total_s": float(time.perf_counter() - started)})
        write_results(out, {"status": "blocked", "reason": reason})
        write_run_meta(out, {"status": "blocked", "validation_boundary": "wsl_meep_env_real_mpb"})
        return out

    solver_cfg_payload = dict(cfg["solver"])
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
            raise ValueError(f"dataset {lane} does not expose the required 9 representative k points")
        if not np.all(np.isfinite(bundle.Wh)) or np.any(np.diff(bundle.Wh, axis=-1) < -1.0e-7):
            raise ValueError(f"dataset {lane} Wh failed finite/ascending checks")
    samples_per_lane = int(dict(cfg["selection"])["samples_per_lane"])
    selections = {lane: select_convergence_indices(bundle.structures, samples_per_lane) for lane, bundle in datasets.items()}
    k_indices_by_lane = {lane: representative_k_indices(bundle.k_grid_index) for lane, bundle in datasets.items()}
    jobs: list[MPBWorkerJob] = []
    job_metadata: dict[int, dict[str, Any]] = {}
    for lane in sorted(datasets):
        bundle = datasets[lane]
        k_indices = k_indices_by_lane[lane]
        for dataset_index in selections[lane]:
            structure = bundle.structures[int(dataset_index)]
            for resolution in (96, 128):
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
        arrays = {64: np.asarray(bundle.Wh[selection][:, k_indices, :], dtype=np.float32)}
        for resolution in (96, 128):
            rows = [
                solved[index]
                for index, metadata in sorted(job_metadata.items())
                if metadata["lane"] == lane and metadata["resolution"] == resolution
            ]
            arrays[resolution] = np.stack(rows, axis=0)
        lane_arrays[lane] = arrays
    gates = dict(cfg.get("gates") or {})
    gate_args = {
        "median_64_96_max": float(gates.get("median_mre_64_96_max", 0.005)),
        "median_96_128_max": float(gates.get("median_mre_96_128_max", 0.0025)),
        "p95_max": float(gates.get("p95_mre_max", 0.01)),
    }
    summaries = {lane: compute_convergence_summary(arrays, **gate_args) for lane, arrays in lane_arrays.items()}
    combined_arrays = {
        resolution: np.concatenate([lane_arrays[lane][resolution] for lane in sorted(lane_arrays)], axis=0)
        for resolution in (64, 96, 128)
    }
    summaries["combined"] = compute_convergence_summary(combined_arrays, **gate_args)
    acceptance_eligible = not skip_environment_check and solver_fn is solve_bands_for_full_cell_mpb
    gate_passed = all(bool(summary["passed"]) for summary in summaries.values())
    status = (
        "resolution_convergence_pass"
        if gate_passed and acceptance_eligible
        else "test_solver_smoke_only"
        if gate_passed
        else "blocked"
    )
    np.savez_compressed(
        out / "convergence_bands.npz",
        **{
            f"{lane}_r{resolution}": array
            for lane, arrays in lane_arrays.items()
            for resolution, array in arrays.items()
        },
    )
    write_json(out / "convergence_metrics.json", {"status": status, "gates": gate_args, "lanes": summaries})
    _write_figure(out / "figures" / "resolution_convergence.png", summaries)
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
            "summaries": summaries,
        },
    )
    write_results(
        out,
        {
            "status": status,
            "gate_passed": gate_passed,
            "next_gate": "publish_combined_manifest" if status == "resolution_convergence_pass" else "inspect_resolution_failures",
        },
    )
    write_run_meta(
        out,
        {
            "status": status,
            "validation_boundary": "wsl_meep_env_real_mpb" if acceptance_eligible else "unaccepted_test_only",
            "config_path": _relative(config_path),
        },
    )
    logger.info("Full-cell resolution convergence completed: %s", status)
    return out


__all__ = [
    "compute_convergence_summary",
    "representative_k_indices",
    "run_full_cell_resolution_convergence",
    "select_convergence_indices",
]
