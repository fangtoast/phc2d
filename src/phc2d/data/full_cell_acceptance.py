"""Cross-environment WSL acceptance for native-Linux full-cell datasets.

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from phc2d.data.full_cell import FullCellDatasetBundle, FullCellStructureSpec, load_full_cell_hdf5
from phc2d.data.full_cell_generation import _solver_config, _validate_execution_boundary
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


AcceptanceSolverFn = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _binary_stratum(structure: FullCellStructureSpec) -> str:
    if structure.generator_family == "unified_fourier_binary":
        return str(structure.metadata.get("stratum", "binary_fourier_unlabelled"))
    if structure.generator_family == "binary_spectral_threshold":
        return "spectral_threshold"
    if structure.generator_family == "binary_polar_holes":
        return f"polar_hole_{int(structure.parameters.get('hole_count', len(structure.parameters.get('holes', []))))}"
    raise ValueError(f"WSL cluster acceptance supports only binary structures, got {structure.generator_family}")


def _graded_stratum(structure: FullCellStructureSpec) -> str:
    if structure.generator_family == "unified_fourier_graded":
        return str(structure.metadata.get("stratum", "graded_fourier_unlabelled"))
    if structure.generator_family != "graded_rbf_fourier":
        raise ValueError(f"graded acceptance requires graded_rbf_fourier, got {structure.generator_family}")
    core_count = int(structure.metadata.get("core_count", len(structure.parameters.get("blobs", []))))
    if core_count not in {1, 2}:
        raise ValueError(f"graded acceptance requires one or two cores, got {core_count}")
    return f"graded_core_{core_count}"


def _stratum(structure: FullCellStructureSpec, lane: str) -> str:
    if lane in {"binary_air_holes", "binary_fourier"}:
        return _binary_stratum(structure)
    if lane in {"graded_medium", "graded_fourier"}:
        return _graded_stratum(structure)
    raise ValueError(f"unsupported full-cell acceptance lane: {lane}")


def select_fixed_structures(
    bundle: FullCellDatasetBundle,
    *,
    lane: str,
    count: int = 8,
) -> list[int]:
    """Select a stable hash-ordered set covering every lane stratum."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, structure in enumerate(bundle.structures):
        grouped[_stratum(structure, lane)].append(index)
    fourier_lane = lane in {"binary_fourier", "graded_fourier"}
    required = (
        {"polar_hole_1", "polar_hole_2", "spectral_threshold"}
        if lane == "binary_air_holes"
        else {"graded_core_1", "graded_core_2"}
    )
    if fourier_lane:
        if not grouped:
            raise ValueError(f"{lane} has no strata")
        ordered_groups = {key: sorted(values, key=lambda index: hashlib.sha256(bundle.structures[index].sample_id.encode()).hexdigest()) for key, values in grouped.items()}
        selected: list[int] = []
        while len(selected) < int(count):
            progress = False
            for key in sorted(ordered_groups):
                if ordered_groups[key] and len(selected) < int(count):
                    selected.append(ordered_groups[key].pop(0))
                    progress = True
            if not progress:
                break
        return sorted(selected)
    if set(grouped) != required:
        raise ValueError(f"{lane} strata mismatch: expected {sorted(required)}, got {sorted(grouped)}")
    if int(count) < len(required):
        raise ValueError("acceptance structure count cannot be smaller than the stratum count")
    ordered = {
        key: sorted(values, key=lambda index: hashlib.sha256(bundle.structures[index].sample_id.encode()).hexdigest())
        for key, values in grouped.items()
    }
    selected = [ordered[key].pop(0) for key in sorted(required)]
    while len(selected) < int(count):
        candidates = [
            (hashlib.sha256(bundle.structures[index].sample_id.encode()).hexdigest(), key, index)
            for key, values in ordered.items()
            for index in values
        ]
        if not candidates:
            raise ValueError("dataset contains fewer structures than requested for acceptance")
        _, key, index = min(candidates)
        ordered[key].remove(index)
        selected.append(index)
    return sorted(selected)


def select_fixed_binary_structures(bundle: FullCellDatasetBundle, *, count: int = 8) -> list[int]:
    """Backward-compatible binary structure selector."""

    return select_fixed_structures(bundle, lane="binary_air_holes", count=count)


def select_representative_k_points(bundle: FullCellDatasetBundle) -> list[int]:
    """Select the center, edge midpoints, and corners from a centered 9x9 grid."""

    desired = {(x, y) for x in (-4, 0, 4) for y in (-4, 0, 4)}
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(bundle.k_grid_index)}
    missing = sorted(desired - set(lookup))
    if missing:
        raise ValueError(f"dataset does not contain the required 9x9 representative k points: {missing}")
    return [lookup[key] for key in sorted(desired)]


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("acceptance config must be a mapping")
    return dict(payload)


def run_cluster_wsl_acceptance(
    dataset_path: str | Path,
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    solver_fn: AcceptanceSolverFn = solve_bands_for_full_cell_mpb,
    skip_environment_check: bool = False,
) -> Path:
    """Recompute fixed full-cell structures in WSL and gate a cluster dataset."""

    dataset = Path(dataset_path).resolve()
    config_file = Path(config_path).resolve()
    cfg = _load_config(config_file)
    acceptance = dict(cfg.get("acceptance") or {})
    lane = str(acceptance.get("lane", "binary_air_holes"))
    solver = dict(cfg.get("solver") or {})
    problems, profile = _validate_execution_boundary(solver)
    if skip_environment_check:
        profile["formal_acceptance_eligible"] = False
        profile["test_only_environment_bypass"] = True
        problems = []
    if problems:
        raise RuntimeError("; ".join(problems))
    if solver.get("environment") != "wsl_meep_env":
        raise ValueError("cross-environment acceptance must use solver.environment=wsl_meep_env")

    task = str((cfg.get("experiment") or {}).get("experiment_id", "cluster_wsl_acceptance_v1"))
    out = make_run_dir(task, runs_dir=str((cfg.get("output") or {}).get("runs_dir", "runs")), run_dir=run_dir)
    setup_logging(log_file=out / "run.log")
    ensure_figures_dir(out)
    write_resolved_config(out, config_file.read_text(encoding="utf-8"))
    write_run_meta(out, {"task": task, "dataset_path": str(dataset), "execution_profile": profile})

    bundle = load_full_cell_hdf5(dataset)
    structure_indices = select_fixed_structures(
        bundle,
        lane=lane,
        count=int(acceptance.get("structure_count", 8)),
    )
    k_indices = select_representative_k_points(bundle)
    k_points = bundle.k_points[k_indices]
    solver_config = _solver_config(solver)
    jobs = [
        MPBWorkerJob(
            sample_id=bundle.structures[index].sample_id,
            sample_index=index,
            split="acceptance",
            rho=np.empty((0,), dtype=np.float32),
            k_points=k_points,
            solver_config=solver_config,
            geometry_hash=bundle.structures[index].geometry_hash,
            solver_input=bundle.structures[index],
        )
        for index in structure_indices
    ]
    worker_cfg = MPBParallelConfig(
        workers=int(acceptance.get("workers", 4)),
        min_workers=1,
        max_workers=int(acceptance.get("workers", 4)),
        worker_backend="processpool",
        fail_fast=False,
        suppress_solver_output=True,
        worker_thread_env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    started = time.perf_counter()
    worker_results, worker_summary = run_mpb_worker_jobs(
        jobs,
        config=worker_cfg,
        shard_dir=out / "worker_shards",
        solver_fn=solver_fn,
    )
    elapsed = time.perf_counter() - started
    failures = [result for result in worker_results if result.status != "computed" or result.bands is None]
    finite_and_ordered = not failures and all(
        np.isfinite(result.bands).all() and np.all(np.diff(result.bands, axis=1) >= -1.0e-7)
        for result in worker_results
        if result.bands is not None
    )
    errors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for result in worker_results:
        if result.bands is None:
            continue
        reference = bundle.Wh[result.sample_index][k_indices, 1:]
        replay = np.asarray(result.bands)[:, 1:]
        relative = np.abs(replay - reference) / np.maximum(np.abs(reference), 1.0e-12)
        errors.append(relative.reshape(-1))
        rows.append(
            {
                "sample_id": result.sample_id,
                "stratum": _stratum(bundle.structures[result.sample_index], lane),
                "median_mre": float(np.median(relative)),
                "p95_mre": float(np.percentile(relative, 95)),
                "solve_time_s": float(result.solve_time_s),
            }
        )
    combined = np.concatenate(errors) if errors else np.asarray([np.inf])
    median_mre = float(np.median(combined))
    p95_mre = float(np.percentile(combined, 95))
    median_limit = float(acceptance.get("median_mre_max", 0.005))
    p95_limit = float(acceptance.get("p95_mre_max", 0.01))
    passed = not failures and finite_and_ordered and median_mre <= median_limit and p95_mre <= p95_limit
    status = "wsl_cross_environment_acceptance_pass" if passed else "pending_wsl_acceptance"
    if skip_environment_check:
        status = "test_only_not_acceptance_eligible"

    with (out / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "stratum", "median_mre", "p95_mre", "solve_time_s"])
        writer.writeheader()
        writer.writerows(rows)
    evidence = {
        "status": status,
        "lane": lane,
        "dataset_path": str(dataset),
        "dataset_sample_count": len(bundle.structures),
        "selected_structure_indices": structure_indices,
        "selected_sample_ids": [bundle.structures[index].sample_id for index in structure_indices],
        "selected_strata": [_stratum(bundle.structures[index], lane) for index in structure_indices],
        "selected_k_grid_indices": bundle.k_grid_index[k_indices].tolist(),
        "bands_compared": list(range(2, int(solver["num_bands"]) + 1)),
        "median_mre": median_mre,
        "p95_mre": p95_mre,
        "median_mre_max": median_limit,
        "p95_mre_max": p95_limit,
        "failure_count": len(failures),
        "finite_and_band_ordered": finite_and_ordered,
        "execution_profile": profile,
    }
    write_json(out / "cross_environment_acceptance.json", evidence)
    write_metrics(
        out,
        {
            "wall_time_total_s": elapsed,
            "replay_structure_count": len(structure_indices),
            "replay_k_point_count": len(k_indices),
            "median_mre": median_mre,
            "p95_mre": p95_mre,
            **worker_summary,
        },
    )
    write_results(out, evidence)
    return out


def run_xingyi_binary_wsl_acceptance(
    dataset_path: str | Path,
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    solver_fn: AcceptanceSolverFn = solve_bands_for_full_cell_mpb,
    skip_environment_check: bool = False,
) -> Path:
    """Backward-compatible wrapper for the Xingyi binary acceptance entrypoint."""

    return run_cluster_wsl_acceptance(
        dataset_path,
        config_path,
        run_dir=run_dir,
        solver_fn=solver_fn,
        skip_environment_check=skip_environment_check,
    )


__all__ = [
    "run_cluster_wsl_acceptance",
    "run_xingyi_binary_wsl_acceptance",
    "select_fixed_binary_structures",
    "select_fixed_structures",
    "select_representative_k_points",
]
