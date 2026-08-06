"""Evidence-gated full-cell dataset generation with real MPB.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from phc2d.data.full_cell import FullCellStructureSpec, build_full_bz_grid, write_full_cell_hdf5
from phc2d.generators.full_cell import (
    _build_spec,
    full_cell_structure_stats,
    generate_binary_polar_holes,
    generate_binary_spectral_threshold,
    generate_graded_rbf_fourier,
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
from phc2d.solvers import (
    FullCellMpbBandSolverConfig,
    MPBParallelConfig,
    MPBWorkerJob,
    MpbBandSolverConfig,
    run_mpb_worker_jobs,
    solve_bands_for_full_cell_mpb,
    solve_circular_air_hole_mpb,
    solve_uniform_medium_mpb,
)


ROOT = Path(__file__).resolve().parents[3]
FullCellSolverFn = Callable[..., tuple[np.ndarray, dict[str, Any]]]
PlanProgressFn = Callable[[int, int], None]
SUPPORTED_EXECUTION_ENVIRONMENTS = {"wsl_meep_env", "native_linux_conda_meep_env"}


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


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


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_full_cell_generation_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Full-cell generation config must be a mapping: {path}")
    return dict(payload)


def _is_wsl() -> bool:
    return "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME"))


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _runtime_version(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ".".join(str(item) for item in value)
    return str(value)


def _execution_profile(solver: Mapping[str, Any]) -> dict[str, Any]:
    environment = str(solver.get("environment", ""))
    profile: dict[str, Any] = {
        "configured_environment": environment,
        "hostname": socket.gethostname(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "libc_name": platform.libc_ver()[0] or "unknown",
        "libc_version": platform.libc_ver()[1] or "unknown",
        "cpu_count": int(os.cpu_count() or 1),
        "cpu_model": _cpu_model(),
        "cluster_id": os.environ.get("PHC2D_CLUSTER_ID", ""),
        "deployed_git_commit": os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT", ""),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE", ""),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
        "slurm_mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU", ""),
    }
    try:
        import meep
        from meep import mpb

        profile["meep_version"] = _runtime_version(getattr(meep, "__version__", "unknown"))
        profile["mpb_version"] = _runtime_version(getattr(mpb, "__version__", "unknown"))
    except Exception as exc:
        profile["meep_mpb_import_error"] = str(exc)
    if environment == "wsl_meep_env":
        profile["validation_boundary"] = "wsl_meep_env_real_mpb"
        profile["formal_acceptance_eligible"] = True
    elif environment == "native_linux_conda_meep_env":
        profile["validation_boundary"] = "native_linux_real_mpb_generation_pending_wsl_acceptance"
        profile["formal_acceptance_eligible"] = False
    else:
        profile["validation_boundary"] = "unsupported_real_mpb_environment"
        profile["formal_acceptance_eligible"] = False
    return profile


def _validate_execution_boundary(solver: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    profile = _execution_profile(solver)
    problems: list[str] = []
    environment = str(solver.get("environment", ""))
    if environment == "wsl_meep_env":
        if not _is_wsl():
            problems.append("wsl_meep_env requires WSL Ubuntu")
        if os.environ.get("CONDA_DEFAULT_ENV", "") != "meep_env":
            problems.append("wsl_meep_env requires conda environment meep_env")
    elif environment == "native_linux_conda_meep_env":
        if platform.system() != "Linux" or _is_wsl():
            problems.append("native_linux_conda_meep_env requires native Linux, not WSL")
        if not os.environ.get("CONDA_PREFIX"):
            problems.append("native_linux_conda_meep_env requires an activated conda environment")
        if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
            problems.append("native_linux_conda_meep_env generation must run inside a Slurm allocation")
    else:
        problems.append(f"unsupported solver.environment: {environment}")
    if profile.get("meep_mpb_import_error"):
        problems.append(f"Meep/MPB Python bindings unavailable: {profile['meep_mpb_import_error']}")
    expected_meep = str(solver.get("expected_meep_version", "")).strip()
    expected_mpb = str(solver.get("expected_mpb_version", "")).strip()
    if expected_meep and profile.get("meep_version") != expected_meep:
        problems.append(f"Meep version {profile.get('meep_version')} != expected {expected_meep}")
    if expected_mpb and profile.get("mpb_version") != expected_mpb:
        problems.append(f"MPB version {profile.get('mpb_version')} != expected {expected_mpb}")
    return problems, profile


def _solver_config(cfg: Mapping[str, Any], *, resolution_override: int | None = None) -> FullCellMpbBandSolverConfig:
    return FullCellMpbBandSolverConfig(
        num_bands=int(cfg["num_bands"]),
        resolution=int(resolution_override if resolution_override is not None else cfg["resolution"]),
        polarization=str(cfg["polarization"]),
        epsilon_air=float(cfg["epsilon_air"]),
        epsilon_material=float(cfg["epsilon_material"]),
        lattice_constant=float(cfg.get("lattice_constant", 1.0)),
        eigensolver_tolerance=float(cfg["eigensolver_tolerance"]),
        band_guard_count=int(cfg.get("band_guard_count", 0)),
        band_order_policy=str(cfg.get("band_order_policy", "frequency_ascending")),
        mesh_size=int(cfg.get("mesh_size", 3)),
    )


def _validate_config(cfg: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    experiment = dict(cfg.get("experiment") or {})
    dataset = dict(cfg.get("dataset") or {})
    generator = dict(cfg.get("generator") or {})
    solver = dict(cfg.get("solver") or {})
    k_grid = dict(cfg.get("k_grid") or {})
    stage = str(experiment.get("stage", "dataset_generation"))
    if not experiment.get("experiment_id"):
        problems.append("experiment.experiment_id is required")
    if stage not in {"solver_preflight", "dataset_generation"}:
        problems.append("experiment.stage must be solver_preflight or dataset_generation")
    if stage == "dataset_generation":
        if dataset.get("protocol") != "full_cell_v1":
            problems.append("dataset.protocol must be full_cell_v1")
        if dataset.get("lane") not in {"graded_medium", "binary_air_holes"}:
            problems.append("dataset.lane must be graded_medium or binary_air_holes")
        if int(dataset.get("sample_count", 0)) <= 0:
            problems.append("dataset.sample_count must be positive")
        split = dict(dataset.get("split") or {})
        if sum(int(split.get(name, 0)) for name in ("train", "val", "test")) != int(dataset.get("sample_count", 0)):
            problems.append("dataset split counts must sum to sample_count")
        if not dataset.get("output_path"):
            problems.append("dataset.output_path is required")
        if [int(value) for value in dataset.get("raster_resolutions", [])] != [32, 64, 128]:
            problems.append("dataset.raster_resolutions must be exactly [32, 64, 128]")
        plan_workers = generator.get("plan_workers", 1)
        if not (isinstance(plan_workers, str) and plan_workers.lower() == "auto"):
            try:
                if int(plan_workers) <= 0:
                    problems.append("generator.plan_workers must be positive or 'auto'")
            except (TypeError, ValueError):
                problems.append("generator.plan_workers must be positive or 'auto'")
    required_solver = {
        "backend",
        "environment",
        "epsilon_air",
        "epsilon_material",
        "resolution",
        "polarization",
        "num_bands",
        "band_guard_count",
        "eigensolver_tolerance",
    }
    missing = sorted(required_solver - set(solver))
    if missing:
        problems.append("missing solver fields: " + ", ".join(missing))
    if str(solver.get("backend", "")).lower() != "mpb" or bool(solver.get("use_proxy", False)):
        problems.append("solver must be real MPB with use_proxy=false")
    if str(solver.get("environment", "")) not in SUPPORTED_EXECUTION_ENVIRONMENTS:
        problems.append(
            "solver.environment must be wsl_meep_env or native_linux_conda_meep_env"
        )
    if str(solver.get("polarization", "")).upper() != "TE":
        problems.append("full_cell_v1 pilot is frozen to TE")
    frozen_solver = {
        "resolution": 64,
        "mesh_size": 3,
        "num_bands": 10,
        "band_guard_count": 4,
    }
    for key, expected in frozen_solver.items():
        if int(solver.get(key, -1)) != expected:
            problems.append(f"solver.{key} must be {expected} for full_cell_v1")
    if not np.isclose(float(solver.get("eigensolver_tolerance", np.nan)), 1.0e-7):
        problems.append("solver.eigensolver_tolerance must be 1e-7 for full_cell_v1")
    if str(solver.get("band_order_policy", "")) != "frequency_ascending":
        problems.append("solver.band_order_policy must be frequency_ascending")
    if stage == "dataset_generation":
        generator = dict(cfg.get("generator") or {})
        for key in ("epsilon_air", "epsilon_material"):
            if not np.isclose(float(generator.get(key, np.nan)), float(solver.get(key, np.nan))):
                problems.append(f"generator.{key} must match solver.{key}")
    try:
        n_side = int(k_grid.get("n_side", 0))
        build_full_bz_grid(n_side)
        if n_side != 9:
            problems.append("k_grid.n_side must be 9 for full_cell_v1")
    except (TypeError, ValueError) as exc:
        problems.append(f"invalid k_grid.n_side: {exc}")
    if str(k_grid.get("coordinate_system", "")) != "mpb_reduced_reciprocal_coordinates":
        problems.append("k_grid coordinate_system must be mpb_reduced_reciprocal_coordinates")
    if str(k_grid.get("family", "")) != "square_full_bz":
        problems.append("k_grid.family must be square_full_bz")
    if str(k_grid.get("endpoint_policy", "")) != "centered_unique_no_boundary_duplicates":
        problems.append("k_grid.endpoint_policy must be centered_unique_no_boundary_duplicates")
    return problems


def _effective_split_counts(split: Mapping[str, Any], full_count: int, effective_count: int) -> dict[str, int]:
    if int(effective_count) == int(full_count):
        return {name: int(split[name]) for name in ("train", "val", "test")}
    exact = {name: int(effective_count) * int(split[name]) / max(int(full_count), 1) for name in ("train", "val", "test")}
    result = {name: int(np.floor(value)) for name, value in exact.items()}
    for name in sorted(exact, key=lambda item: exact[item] - result[item], reverse=True)[: int(effective_count) - sum(result.values())]:
        result[name] += 1
    return result


def _stable_stratified_splits(
    sample_ids: Sequence[str],
    strata: Sequence[str],
    split_counts: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if len(sample_ids) != len(strata):
        raise ValueError("sample_ids and strata must have equal length")
    names = ("train", "val", "test")
    total = len(sample_ids)
    if sum(int(split_counts[name]) for name in names) != total:
        raise ValueError("split counts must sum to sample count")
    fractions = {name: int(split_counts[name]) / max(total, 1) for name in names}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[str(stratum)].append(index)
    labels = [""] * total
    assigned = Counter()
    for stratum in sorted(grouped):
        indices = sorted(grouped[stratum], key=lambda index: hashlib.sha256(sample_ids[index].encode("utf-8")).hexdigest())
        exact = {name: len(indices) * fractions[name] for name in names}
        counts = {name: int(np.floor(exact[name])) for name in names}
        remaining = len(indices) - sum(counts.values())
        for name in sorted(names, key=lambda item: (exact[item] - counts[item], -names.index(item)), reverse=True)[:remaining]:
            counts[name] += 1
        cursor = 0
        for name in names:
            for index in indices[cursor : cursor + counts[name]]:
                labels[index] = name
            cursor += counts[name]
            assigned[name] += counts[name]
    # Reconcile rare largest-remainder drift while preserving deterministic ordering.
    for target in names:
        while assigned[target] < int(split_counts[target]):
            donor = next(name for name in names if assigned[name] > int(split_counts[name]))
            candidates = [index for index, label_name in enumerate(labels) if label_name == donor]
            move = min(candidates, key=lambda index: hashlib.sha256(f"move:{target}:{sample_ids[index]}".encode("utf-8")).hexdigest())
            labels[move] = target
            assigned[donor] -= 1
            assigned[target] += 1
    arrays = tuple(np.flatnonzero(np.asarray(labels, dtype=object) == name).astype(np.int64) for name in names)
    return arrays[0], arrays[1], arrays[2], labels


def _structure_schedule(lane: str, count: int) -> list[tuple[str, int | None, str]]:
    rows: list[tuple[str, int | None, str]] = []
    if lane == "graded_medium":
        for index in range(count):
            core_count = 1 if index % 2 == 0 else 2
            rows.append(("graded_rbf_fourier", core_count, f"graded_core_{core_count}"))
        return rows
    cycle = (
        ("binary_polar_holes", 1, "polar_hole_1"),
        ("binary_polar_holes", 2, "polar_hole_2"),
        ("binary_spectral_threshold", None, "spectral_threshold"),
        ("binary_spectral_threshold", None, "spectral_threshold"),
    )
    return [cycle[index % len(cycle)] for index in range(count)]


def _generate_structure_task(
    task: tuple[int, str, int | None, str, int, str, dict[str, Any]],
) -> tuple[FullCellStructureSpec, str]:
    index, family, variant, stratum, seed, experiment_id, generator_cfg = task
    sample_id = f"{experiment_id}_{index:06d}"
    if family == "graded_rbf_fourier":
        structure = generate_graded_rbf_fourier(
            sample_id,
            master_seed=seed,
            core_count=int(variant),
            config=generator_cfg,
        )
    elif family == "binary_polar_holes":
        structure = generate_binary_polar_holes(
            sample_id,
            master_seed=seed,
            hole_count=int(variant),
            config=generator_cfg,
        )
    else:
        structure = generate_binary_spectral_threshold(
            sample_id,
            master_seed=seed,
            config=generator_cfg,
        )
    return structure, stratum


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, int(os.cpu_count() or 1))


def _resolve_plan_workers(
    cfg: Mapping[str, Any],
    *,
    effective_count: int,
    workers: int | str | None,
) -> int:
    requested: int | str = workers if workers is not None else dict(cfg.get("generator") or {}).get("plan_workers", 1)
    available = _available_cpu_count()
    if isinstance(requested, str) and requested.lower() == "auto":
        return max(1, min(available, effective_count))
    try:
        resolved = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("plan workers must be a positive integer or 'auto'") from exc
    if resolved <= 0:
        raise ValueError("plan workers must be positive")
    if resolved > available:
        raise RuntimeError(
            f"plan workers requested {resolved}, but only {available} CPUs are available"
        )
    return max(1, min(resolved, effective_count))


def _generate_structures(
    cfg: Mapping[str, Any],
    *,
    effective_count: int,
    workers: int = 1,
    progress_fn: PlanProgressFn | None = None,
) -> tuple[list[FullCellStructureSpec], list[str]]:
    dataset = dict(cfg["dataset"])
    generator_cfg = dict(cfg.get("generator") or {})
    lane = str(dataset["lane"])
    seed = int(dataset["seed"])
    experiment_id = str(dict(cfg["experiment"])["experiment_id"])
    tasks = [
        (index, family, variant, stratum, seed, experiment_id, generator_cfg)
        for index, (family, variant, stratum) in enumerate(_structure_schedule(lane, effective_count))
    ]
    resolved_workers = max(1, min(int(workers), max(effective_count, 1)))
    if progress_fn is not None:
        progress_fn(0, effective_count)
    if resolved_workers == 1:
        generated = map(_generate_structure_task, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=resolved_workers,
            mp_context=get_context("spawn"),
        )
        chunksize = max(1, effective_count // (resolved_workers * 8))
        generated = pool.map(_generate_structure_task, tasks, chunksize=chunksize)
    structures: list[FullCellStructureSpec] = []
    strata: list[str] = []
    try:
        for completed, (structure, stratum) in enumerate(generated, start=1):
            structures.append(structure)
            strata.append(stratum)
            if progress_fn is not None:
                progress_fn(completed, effective_count)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return structures, strata


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    _write_jsonl(temporary, rows)
    temporary.replace(path)


def _plan_progress_writer(path: Path, *, workers: int, started: float) -> PlanProgressFn:
    last_written = -1

    def report(completed: int, total: int) -> None:
        nonlocal last_written
        interval = max(1, total // 100)
        if completed not in {0, total} and completed - last_written < interval:
            return
        last_written = completed
        elapsed = float(time.perf_counter() - started)
        payload = {
            "status": "completed" if completed == total else "running",
            "completed": int(completed),
            "total": int(total),
            "workers": int(workers),
            "elapsed_s": elapsed,
            "throughput_samples_per_s": float(completed / elapsed) if elapsed > 0.0 else 0.0,
        }
        temporary = path.with_name(f".{path.name}.tmp")
        write_json(temporary, payload)
        temporary.replace(path)

    return report


def _load_frozen_sample_plan(
    path: Path,
    cfg: Mapping[str, Any],
    *,
    effective_count: int,
) -> tuple[list[FullCellStructureSpec], list[str], list[str]]:
    """Reconstruct a validated structure schedule without resampling geometry."""

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != effective_count:
        raise ValueError(f"Frozen sample plan has {len(rows)} rows; expected {effective_count}")
    experiment_id = str(dict(cfg["experiment"])["experiment_id"])
    generator_cfg = dict(cfg.get("generator") or {})
    epsilon_air = float(generator_cfg.get("epsilon_air", 1.0))
    epsilon_material = float(generator_cfg.get("epsilon_material", 8.9))
    expected_schedule = _structure_schedule(str(dict(cfg["dataset"])["lane"]), effective_count)
    structures: list[FullCellStructureSpec] = []
    strata: list[str] = []
    splits: list[str] = []
    for index, (row, (expected_family, _variant, expected_stratum)) in enumerate(zip(rows, expected_schedule)):
        sample_id = f"{experiment_id}_{index:06d}"
        if int(row.get("sample_index", -1)) != index or str(row.get("sample_id")) != sample_id:
            raise ValueError(f"Frozen sample plan identity mismatch at row {index}")
        family = str(row.get("generator_family"))
        stratum = str(row.get("stratum"))
        if family != expected_family or stratum != expected_stratum:
            raise ValueError(f"Frozen sample plan schedule mismatch at row {index}")
        split = str(row.get("split"))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Frozen sample plan split mismatch at row {index}")
        structure = _build_spec(
            sample_id=sample_id,
            family=family,
            material_kind="continuous" if family == "graded_rbf_fourier" else "binary",
            payload=dict(row.get("parameters") or {}),
            metadata={"stats": dict(row.get("stats") or {}), "resumed_from_sample_plan": True},
            epsilon_air=epsilon_air,
            epsilon_material=epsilon_material,
        )
        if structure.geometry_hash != str(row.get("geometry_hash")):
            raise ValueError(f"Frozen sample plan geometry hash mismatch at row {index}")
        structures.append(structure)
        strata.append(stratum)
        splits.append(split)
    expected_ids = [item.sample_id for item in structures]
    split_counts = _effective_split_counts(dict(dict(cfg["dataset"])["split"]), int(dict(cfg["dataset"])["sample_count"]), effective_count)
    _, _, _, expected_splits = _stable_stratified_splits(expected_ids, strata, split_counts)
    if splits != expected_splits:
        raise ValueError("Frozen sample plan split assignments do not match the configured stable split")
    return structures, strata, splits


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "status", "reason"])
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in writer.fieldnames} for row in rows)


def prepare_full_cell_sample_plan(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    workers: int | str | None = None,
) -> Path:
    """Generate or validate a frozen full-cell structure plan without invoking MPB."""

    config_path = _repo_path(config_path)
    cfg = load_full_cell_generation_config(config_path)
    experiment = dict(cfg.get("experiment") or {})
    experiment_id = str(experiment.get("experiment_id", "full_cell_generation"))
    output_cfg = dict(cfg.get("output") or {})
    out = make_run_dir(
        f"{experiment_id}_plan",
        runs_dir=_repo_path(output_cfg.get("runs_dir", "runs")),
        run_dir=_repo_path(run_dir) if run_dir else None,
    )
    logger = setup_logging(
        log_file=out / "run.log",
        run_meta={
            "script": "phc2d.data.full_cell_generation.prepare_full_cell_sample_plan",
            "config_path": _relative(config_path),
        },
    )
    started = time.perf_counter()
    resolved_yaml = yaml.safe_dump(
        {**cfg, "run_options": {"plan_only": True, "plan_workers_override": workers}},
        sort_keys=False,
        allow_unicode=True,
    )
    write_resolved_config(out, resolved_yaml)
    problems = _validate_config(cfg)
    if str(experiment.get("stage")) != "dataset_generation":
        problems.append("sample-plan preparation requires experiment.stage=dataset_generation")
    if problems:
        reason = "; ".join(problems)
        _write_history(
            out / "history.csv",
            [{"stage": "sample_plan", "status": "blocked", "reason": reason}],
        )
        write_metrics(
            out,
            {"status": "blocked", "wall_time_total_s": float(time.perf_counter() - started)},
        )
        write_results(out, {"status": "blocked", "reason": reason, "sample_plan_written": False})
        write_run_meta(
            out,
            {"status": "blocked", "validation_boundary": "structure_plan_only_no_mpb"},
        )
        return out

    dataset_cfg = dict(cfg["dataset"])
    sample_count = int(dataset_cfg["sample_count"])
    split_counts = {
        name: int(dict(dataset_cfg["split"])[name]) for name in ("train", "val", "test")
    }
    sample_plan_path = out / "sample_plan.jsonl"
    reused = sample_plan_path.is_file()
    plan_workers = 0
    if reused:
        structures, strata, split_labels = _load_frozen_sample_plan(
            sample_plan_path,
            cfg,
            effective_count=sample_count,
        )
    else:
        plan_workers = _resolve_plan_workers(cfg, effective_count=sample_count, workers=workers)
        plan_started = time.perf_counter()
        progress_path = out / "sample_plan_progress.json"
        progress_fn = _plan_progress_writer(progress_path, workers=plan_workers, started=plan_started)
        structures, strata = _generate_structures(
            cfg,
            effective_count=sample_count,
            workers=plan_workers,
            progress_fn=progress_fn,
        )
        sample_ids = [structure.sample_id for structure in structures]
        _, _, _, split_labels = _stable_stratified_splits(sample_ids, strata, split_counts)
        rows = [
            {
                "sample_id": structure.sample_id,
                "sample_index": index,
                "split": split_labels[index],
                "stratum": strata[index],
                "generator_family": structure.generator_family,
                "geometry_hash": structure.geometry_hash,
                "parameters": structure.parameters,
                "stats": structure.metadata.get("stats", {}),
            }
            for index, structure in enumerate(structures)
        ]
        _write_jsonl_atomic(sample_plan_path, rows)

    plan_hash = _file_hash(sample_plan_path)
    manifest = {
        "schema_version": "phc2d.full_cell_sample_plan.v1",
        "status": "full_cell_sample_plan_ready",
        "validation_boundary": "structure_plan_only_no_mpb",
        "config_path": _relative(config_path),
        "config_hash": _hash_text(resolved_yaml),
        "sample_plan_path": _relative(sample_plan_path),
        "sample_plan_sha256": plan_hash,
        "sample_count": sample_count,
        "lane": dataset_cfg["lane"],
        "split": split_counts,
        "strata": dict(Counter(strata)),
        "reused_existing_plan": reused,
        "plan_workers_resolved": plan_workers,
    }
    write_json(out / "sample_plan_manifest.json", manifest)
    elapsed = float(time.perf_counter() - started)
    _write_history(
        out / "history.csv",
        [{"stage": "sample_plan", "status": "reused" if reused else "generated", "reason": ""}],
    )
    write_metrics(
        out,
        {
            "status": "full_cell_sample_plan_ready",
            "wall_time_total_s": elapsed,
            "sample_count": sample_count,
            "stratum_counts": dict(Counter(strata)),
            "plan_workers_resolved": plan_workers,
            "plan_throughput_samples_per_s": float(sample_count / elapsed) if not reused and elapsed > 0.0 else 0.0,
        },
    )
    write_results(
        out,
        {
            "status": "full_cell_sample_plan_ready",
            "sample_plan_written": True,
            "sample_plan_path": _relative(sample_plan_path),
            "sample_plan_sha256": plan_hash,
            "plan_workers_resolved": plan_workers,
            "next_gate": "native_cluster_real_mpb_generation",
        },
    )
    write_run_meta(
        out,
        {
            "status": "full_cell_sample_plan_ready",
            "validation_boundary": "structure_plan_only_no_mpb",
            "sample_plan_sha256": plan_hash,
        },
    )
    logger.info("Full-cell sample plan ready: %s", sample_plan_path)
    return out


def _write_preview_figures(
    figures_dir: Path,
    structures: Sequence[FullCellStructureSpec],
    bands: np.ndarray,
    *,
    n_side: int,
    raster_resolutions: Sequence[int],
) -> None:
    import matplotlib.pyplot as plt

    count = min(6, len(structures))
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), squeeze=False)
    for axis, structure in zip(axes.flat, structures[:count]):
        image = axis.imshow(structure.epsilon_grid((64, 64)), origin="lower", cmap="Blues", vmin=structure.epsilon_air, vmax=structure.epsilon_material)
        axis.set_title(f"{structure.generator_family}\n{structure.sample_id}", fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
    for axis in list(axes.flat)[count:]:
        axis.axis("off")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="epsilon")
    fig.savefig(figures_dir / "sample_gallery.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    resolutions = [int(value) for value in raster_resolutions]
    fig, axes = plt.subplots(1, len(resolutions), figsize=(3 * len(resolutions), 3), squeeze=False)
    for axis, resolution in zip(axes.flat, resolutions):
        axis.imshow(structures[0].epsilon_grid((resolution, resolution)), origin="lower", cmap="Blues")
        axis.set_title(f"{resolution}x{resolution}")
        axis.set_xticks([]); axis.set_yticks([])
    fig.savefig(figures_dir / "resolution_pairing.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    band_index = min(1, bands.shape[-1] - 1)
    fig, axis = plt.subplots(figsize=(4.5, 4))
    heatmap = axis.imshow(bands[0, :, band_index].reshape(n_side, n_side), origin="lower", cmap="viridis")
    axis.set_title(f"sample 0, band {band_index + 1}")
    axis.set_xlabel("k_y index"); axis.set_ylabel("k_x index")
    fig.colorbar(heatmap, ax=axis, label="a/lambda")
    fig.savefig(figures_dir / "bz_band_preview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_full_cell_generation(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    max_samples: int | None = None,
    workers: int | str | None = None,
    plan_workers: int | str | None = None,
    skip_environment_check: bool = False,
    solver_fn: FullCellSolverFn = solve_bands_for_full_cell_mpb,
) -> Path:
    """Generate one full-cell lane or write a blocked run with complete evidence."""

    config_path = _repo_path(config_path)
    cfg = load_full_cell_generation_config(config_path)
    experiment = dict(cfg.get("experiment") or {})
    experiment_id = str(experiment.get("experiment_id", "full_cell_generation"))
    output_cfg = dict(cfg.get("output") or {})
    out = make_run_dir(experiment_id, runs_dir=_repo_path(output_cfg.get("runs_dir", "runs")), run_dir=_repo_path(run_dir) if run_dir else None)
    logger = setup_logging(log_file=out / "run.log", run_meta={"script": "phc2d.data.full_cell_generation", "config_path": _relative(config_path)})
    ensure_figures_dir(out)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    resolved = {
        **cfg,
        "run_options": {
            "max_samples": max_samples,
            "workers_override": workers,
            "plan_workers_override": plan_workers,
        },
    }
    resolved_yaml = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    write_resolved_config(out, resolved_yaml)
    problems = _validate_config(cfg)
    solver_payload = dict(cfg.get("solver") or {})
    execution_profile = _execution_profile(solver_payload)
    if not skip_environment_check:
        environment_problems, execution_profile = _validate_execution_boundary(solver_payload)
        problems.extend(environment_problems)
    if problems:
        reason = "; ".join(problems)
        history.append({"stage": "contract_check", "status": "blocked", "reason": reason})
        _write_history(out / "history.csv", history)
        write_metrics(out, {"status": "blocked", "wall_time_total_s": float(time.perf_counter() - started), "real_mpb_generation_started": False})
        write_results(out, {"status": "blocked", "reason": reason, "dataset_written": False})
        write_run_meta(out, {"status": "blocked", "validation_boundary": execution_profile["validation_boundary"], "execution_profile": execution_profile})
        return out
    if str(experiment.get("stage")) == "solver_preflight":
        return run_full_cell_solver_preflight(
            config_path,
            run_dir=out,
            skip_environment_check=skip_environment_check,
        )

    dataset_cfg = dict(cfg["dataset"])
    full_count = int(dataset_cfg["sample_count"])
    effective_count = min(full_count, int(max_samples)) if max_samples is not None else full_count
    split_counts = _effective_split_counts(dict(dataset_cfg["split"]), full_count, effective_count)
    sample_plan_path = out / "sample_plan.jsonl"
    structure_plan_started = time.perf_counter()
    plan_workers_resolved = 0
    if sample_plan_path.is_file():
        structures, strata, split_labels = _load_frozen_sample_plan(sample_plan_path, cfg, effective_count=effective_count)
        sample_ids = [structure.sample_id for structure in structures]
        idx_train, idx_val, idx_test, expected_labels = _stable_stratified_splits(sample_ids, strata, split_counts)
        if split_labels != expected_labels:
            raise ValueError("Frozen sample plan split assignments changed during resume")
        history.append({"stage": "structure_generation", "status": "resumed", "reason": "validated frozen sample_plan.jsonl"})
    else:
        plan_workers_resolved = _resolve_plan_workers(
            cfg,
            effective_count=effective_count,
            workers=plan_workers,
        )
        progress_fn = _plan_progress_writer(
            out / "sample_plan_progress.json",
            workers=plan_workers_resolved,
            started=structure_plan_started,
        )
        structures, strata = _generate_structures(
            cfg,
            effective_count=effective_count,
            workers=plan_workers_resolved,
            progress_fn=progress_fn,
        )
        sample_ids = [structure.sample_id for structure in structures]
        idx_train, idx_val, idx_test, split_labels = _stable_stratified_splits(sample_ids, strata, split_counts)
        history.append(
            {
                "stage": "structure_generation",
                "status": "passed",
                "reason": f"generated with {plan_workers_resolved} process workers",
            }
        )
    structure_plan_time_s = float(time.perf_counter() - structure_plan_started)
    k_points, k_grid_index = build_full_bz_grid(int(dict(cfg["k_grid"])["n_side"]))
    solver_cfg = _solver_config(dict(cfg["solver"]))
    worker_payload = dict(cfg.get("workers") or {})
    if workers is not None:
        worker_payload["workers"] = workers
    parallel = MPBParallelConfig.from_mapping(worker_payload)
    jobs = [
        MPBWorkerJob(
            sample_id=structure.sample_id,
            sample_index=index,
            split=split_labels[index],
            rho=np.asarray(structure.parameter_vector, dtype=np.float32),
            solver_input=structure,
            k_points=k_points,
            solver_config=solver_cfg,
            rho_hash=_hash_text(_canonical_json(structure.parameter_vector)),
            geometry_hash=structure.geometry_hash,
        )
        for index, structure in enumerate(structures)
    ]
    if not sample_plan_path.is_file():
        _write_jsonl_atomic(
            sample_plan_path,
            [
                {
                    "sample_id": structure.sample_id,
                    "sample_index": index,
                    "split": split_labels[index],
                    "stratum": strata[index],
                    "generator_family": structure.generator_family,
                    "geometry_hash": structure.geometry_hash,
                    "parameters": structure.parameters,
                    "stats": structure.metadata.get("stats", {}),
                }
                for index, structure in enumerate(structures)
            ],
        )
    results, worker_summary = run_mpb_worker_jobs(jobs, config=parallel, shard_dir=out / "worker_shards", solver_fn=solver_fn)
    failures = [
        {"sample_id": item.sample_id, "sample_index": item.sample_index, "status": item.status, "reason": item.error}
        for item in results
        if item.status != "computed" or item.bands is None
    ]
    _write_jsonl(out / "failed_case_ledger.jsonl", failures)
    if failures or len(results) != effective_count:
        reason = failures[0]["reason"] if failures else "worker result count mismatch"
        history.append({"stage": "real_mpb_generation", "status": "solver_execution_failed", "reason": reason})
        _write_history(out / "history.csv", history)
        write_metrics(
            out,
            {
                **worker_summary,
                "status": "solver_execution_failed",
                "wall_time_total_s": float(time.perf_counter() - started),
                "structure_plan_time_s": structure_plan_time_s,
                "plan_workers_resolved": plan_workers_resolved,
                "real_mpb_generation_started": True,
                "real_mpb_generation_completed": False,
            },
        )
        write_results(out, {"status": "solver_execution_failed", "reason": reason, "dataset_written": False})
        write_run_meta(
            out,
            {
                "status": "solver_execution_failed",
                "validation_boundary": execution_profile["validation_boundary"],
                "execution_profile": execution_profile,
            },
        )
        return out
    bands = np.stack([np.asarray(item.bands, dtype=np.float32) for item in results], axis=0)
    expected = (effective_count, len(k_points), int(solver_cfg.num_bands))
    if bands.shape != expected or not np.all(np.isfinite(bands)):
        raise RuntimeError(f"Full-cell solver output failed shape/finite gate: got {bands.shape}, expected {expected}")
    if np.any(np.diff(bands, axis=-1) < -1.0e-7):
        raise RuntimeError("Full-cell solver output violates ascending band order")
    raster_resolutions = tuple(int(value) for value in dataset_cfg["raster_resolutions"])
    configured_output = _repo_path(str(dataset_cfg["output_path"]))
    dataset_path = configured_output if effective_count == full_count else out / f"{configured_output.stem}_smoke_{effective_count}.h5"
    real_mpb_generation_eligible = not skip_environment_check and solver_fn is solve_bands_for_full_cell_mpb
    formal_acceptance_eligible = bool(
        real_mpb_generation_eligible and execution_profile.get("formal_acceptance_eligible")
    )
    validation_boundary = (
        str(execution_profile["validation_boundary"])
        if real_mpb_generation_eligible
        else "unaccepted_test_only_injected_or_unchecked_solver"
    )
    metadata = {
        "protocol": "full_cell_v1",
        "lane": dataset_cfg["lane"],
        "sample_count": effective_count,
        "split": split_counts,
        "strata": dict(Counter(strata)),
        "k_grid": dict(cfg["k_grid"]),
        "solver": dict(cfg["solver"]),
        "config_path": _relative(config_path),
        "config_hash": _hash_text(resolved_yaml),
        "validation_boundary": validation_boundary,
    }
    write_full_cell_hdf5(
        dataset_path,
        structures=structures,
        Wh=bands,
        k_points=k_points,
        k_grid_index=k_grid_index,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        raster_resolutions=raster_resolutions,
        metadata=metadata,
    )
    figures = ensure_figures_dir(out)
    _write_preview_figures(figures, structures, bands, n_side=int(dict(cfg["k_grid"])["n_side"]), raster_resolutions=raster_resolutions)
    sample_rows = []
    for result, structure, stratum in zip(results, structures, strata):
        sample_rows.append(
            {
                "sample_id": structure.sample_id,
                "sample_index": result.sample_index,
                "split": result.split,
                "stratum": stratum,
                "generator_family": structure.generator_family,
                "geometry_hash": structure.geometry_hash,
                "Wh_hash": result.Wh_hash,
                "mpb_solve_time_s": result.solve_time_s,
                "solver_metadata": result.solve_metadata,
                "structure_stats": structure.metadata.get("stats", {}),
            }
        )
    _write_jsonl(out / "sample_metadata.jsonl", sample_rows)
    provenance = {
        "backend": "meep.mpb.ModeSolver" if real_mpb_generation_eligible else f"{solver_fn.__module__}.{solver_fn.__name__}",
        "execution_boundary": validation_boundary,
        "real_mpb_generation_eligible": real_mpb_generation_eligible,
        "acceptance_eligible": formal_acceptance_eligible,
        "execution_profile": execution_profile,
        "geometry_convention": "analytic_periodic_full_cell_material_function",
        "solver": dict(cfg["solver"]),
        "k_grid": {**dict(cfg["k_grid"]), "k_points_hash": _hash_text(_canonical_json(k_points))},
        "dataset_file_hash": _file_hash(dataset_path),
        "worker_summary": worker_summary,
    }
    write_json(out / "solver_provenance.json", provenance)
    native_cluster = str(solver_payload.get("environment")) == "native_linux_conda_meep_env"
    if not real_mpb_generation_eligible:
        status = "test_solver_smoke_only"
    elif native_cluster and effective_count <= 4 and effective_count < full_count:
        status = "cluster_real_mpb_preflight_pass_pending_wsl_acceptance"
    elif native_cluster:
        status = "cluster_generated_pending_wsl_acceptance"
    elif effective_count <= 4 and effective_count < full_count:
        status = "real_mpb_preflight_pass"
    else:
        status = "pilot_generated_pending_convergence"
    manifest = {
        "schema_version": "phc2d.full_cell_manifest.v1",
        "status": status,
        "dataset_path": _relative(dataset_path),
        "dataset_sha256": _file_hash(dataset_path),
        "lane": dataset_cfg["lane"],
        "sample_count": effective_count,
        "Wh_shape": list(bands.shape),
        "k_grid_shape": [int(dict(cfg["k_grid"])["n_side"])] * 2,
        "raster_resolutions": list(raster_resolutions),
        "split": split_counts,
        "solver_provenance": "solver_provenance.json",
        "sample_metadata": "sample_metadata.jsonl",
        "failed_case_ledger": "failed_case_ledger.jsonl",
        "next_gate": (
            "wsl_cross_environment_acceptance"
            if status in {"cluster_real_mpb_preflight_pass_pending_wsl_acceptance", "cluster_generated_pending_wsl_acceptance"}
            else "resolution_convergence_64_96_128"
            if status == "pilot_generated_pending_convergence"
            else "solver_oracle_preflight_then_full_pilot"
            if status == "real_mpb_preflight_pass"
            else "rerun_in_wsl_meep_env_with_real_mpb"
        ),
    }
    (out / "dataset_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    history.append({"stage": "real_mpb_generation", "status": status, "reason": ""})
    _write_history(out / "history.csv", history)
    elapsed = float(time.perf_counter() - started)
    write_metrics(
        out,
        {
            **worker_summary,
            "status": status,
            "wall_time_total_s": elapsed,
            "structure_generation_count": effective_count,
            "structure_plan_time_s": structure_plan_time_s,
            "plan_workers_resolved": plan_workers_resolved,
            "mpb_generation_time_s": float(sum(item.solve_time_s for item in results if not item.resumed)),
            "mpb_generation_time_per_sample_s": float(sum(item.solve_time_s for item in results) / max(effective_count, 1)),
            "real_mpb_generation_started": True,
            "real_mpb_generation_completed": True,
            "solver_failure_rate": 0.0,
            "family_counts": dict(Counter(structure.generator_family for structure in structures)),
            "stratum_counts": dict(Counter(strata)),
        },
    )
    write_results(out, {"status": status, "dataset_written": True, "dataset_path": _relative(dataset_path), "dataset_manifest": "dataset_manifest.yaml", "next_gate": manifest["next_gate"]})
    write_run_meta(out, {"status": status, "validation_boundary": validation_boundary, "config_path": _relative(config_path), "execution_profile": execution_profile})
    logger.info("Full-cell generation completed with status %s: %s", status, dataset_path)
    return out


def _uniform_structure(sample_id: str, *, material_weight: float, solver_cfg: MpbBandSolverConfig) -> FullCellStructureSpec:
    bias = 1000.0 if material_weight <= 0.0 else -1000.0
    return FullCellStructureSpec(
        sample_id=sample_id,
        generator_family="graded_rbf_fourier",
        material_kind="continuous",
        parameters={"blobs": [], "modes": [], "fourier_strength": 0.0, "temperature": 1.0, "bias": bias, "target_material_fraction": float(material_weight)},
        parameter_names=("bias",),
        parameter_vector=(bias,),
        parameter_mask=(True,),
        epsilon_air=solver_cfg.epsilon_air,
        epsilon_material=solver_cfg.epsilon_material,
    )


def _circle_structure(sample_id: str, *, radius: float, solver_cfg: MpbBandSolverConfig) -> FullCellStructureSpec:
    payload = {"hole_count": 1, "holes": [{"center_x": 0.0, "center_y": 0.0, "radius": float(radius), "modes": [{"order": order, "cos": 0.0, "sin": 0.0} for order in range(2, 6)]}]}
    return FullCellStructureSpec(
        sample_id=sample_id,
        generator_family="binary_polar_holes",
        material_kind="binary",
        parameters=payload,
        parameter_names=("radius",),
        parameter_vector=(float(radius),),
        parameter_mask=(True,),
        epsilon_air=solver_cfg.epsilon_air,
        epsilon_material=solver_cfg.epsilon_material,
    )


def _representative_k_points(n_side: int) -> np.ndarray:
    points, indices = build_full_bz_grid(n_side)
    half = n_side // 2
    wanted = {(-half, -half), (-half, 0), (-half, half), (0, -half), (0, 0), (0, half), (half, -half), (half, 0), (half, half)}
    return np.asarray([point for point, index in zip(points, indices) if tuple(int(value) for value in index) in wanted], dtype=np.float64)


def run_full_cell_solver_preflight(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    skip_environment_check: bool = False,
) -> Path:
    """Compare material-function encoding against native MPB reference geometry."""

    config_path = _repo_path(config_path)
    cfg = load_full_cell_generation_config(config_path)
    experiment = dict(cfg["experiment"])
    output_cfg = dict(cfg.get("output") or {})
    out = _repo_path(run_dir) if run_dir is not None else make_run_dir(str(experiment["experiment_id"]), runs_dir=_repo_path(output_cfg.get("runs_dir", "runs")))
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file=out / "run.log", run_meta={"script": "phc2d.data.full_cell_generation", "stage": "solver_preflight"})
    ensure_figures_dir(out)
    started = time.perf_counter()
    resolved_yaml = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    write_resolved_config(out, resolved_yaml)
    problems = _validate_config(cfg)
    solver_payload = dict(cfg.get("solver") or {})
    execution_profile = _execution_profile(solver_payload)
    if not skip_environment_check:
        environment_problems, execution_profile = _validate_execution_boundary(solver_payload)
        problems.extend(environment_problems)
    if problems:
        reason = "; ".join(problems)
        _write_history(out / "history.csv", [{"stage": "solver_preflight", "status": "blocked", "reason": reason}])
        write_metrics(out, {"status": "blocked", "wall_time_total_s": float(time.perf_counter() - started)})
        write_results(out, {"status": "blocked", "reason": reason})
        write_run_meta(out, {"status": "blocked", "validation_boundary": execution_profile["validation_boundary"], "execution_profile": execution_profile})
        return out
    solver_cfg = _solver_config(dict(cfg["solver"]))
    k_points = _representative_k_points(int(dict(cfg["k_grid"])["n_side"]))
    cases: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    def measured(name: str, function: Callable[[], tuple[np.ndarray, dict[str, Any]]]) -> np.ndarray:
        case_started = time.perf_counter()
        values, metadata = function()
        timings[name] = float(time.perf_counter() - case_started)
        write_json(out / f"{name}.metadata.json", metadata)
        cases[name] = np.asarray(values, dtype=np.float64)
        return cases[name]

    air_structure = _uniform_structure("oracle_uniform_air", material_weight=0.0, solver_cfg=solver_cfg)
    material_structure = _uniform_structure("oracle_uniform_material", material_weight=1.0, solver_cfg=solver_cfg)
    radius = float(dict(cfg.get("preflight") or {}).get("circle_radius", 0.20))
    circle_structure = _circle_structure("oracle_circle", radius=radius, solver_cfg=solver_cfg)
    measured("uniform_air_material_function", lambda: solve_bands_for_full_cell_mpb(air_structure, k_points=k_points, config=solver_cfg))
    measured("uniform_air_native", lambda: solve_uniform_medium_mpb(solver_cfg.epsilon_air, k_points=k_points, config=solver_cfg))
    measured("uniform_material_function", lambda: solve_bands_for_full_cell_mpb(material_structure, k_points=k_points, config=solver_cfg))
    measured("uniform_material_native", lambda: solve_uniform_medium_mpb(solver_cfg.epsilon_material, k_points=k_points, config=solver_cfg))
    measured("circle_material_function", lambda: solve_bands_for_full_cell_mpb(circle_structure, k_points=k_points, config=solver_cfg))
    measured("circle_material_function_repeat", lambda: solve_bands_for_full_cell_mpb(circle_structure, k_points=k_points, config=solver_cfg))
    measured("circle_native", lambda: solve_circular_air_hole_mpb(center=(0.0, 0.0), radius=radius, k_points=k_points, config=solver_cfg))
    air_abs = np.abs(cases["uniform_air_material_function"] - cases["uniform_air_native"])
    material_abs = np.abs(cases["uniform_material_function"] - cases["uniform_material_native"])
    air_max = float(np.max(air_abs))
    material_max = float(np.max(material_abs))
    air_median = float(np.median(air_abs))
    material_median = float(np.median(material_abs))
    repeat_max = float(np.max(np.abs(cases["circle_material_function"] - cases["circle_material_function_repeat"])))
    start_band = 1 if int(solver_cfg.num_bands) > 1 else 0
    reference = cases["circle_native"][:, start_band:]
    circle_mre = float(np.mean(np.abs(cases["circle_material_function"][:, start_band:] - reference) / np.maximum(np.abs(reference), 1.0e-8)))
    gates = dict(cfg.get("preflight") or {})
    uniform_tolerance = float(gates.get("uniform_median_abs_tolerance", 1.0e-6))
    uniform_match_tolerance = float(gates.get("uniform_match_abs_tolerance", 1.0e-5))
    uniform_match_fraction_min = float(gates.get("uniform_match_fraction_min", 0.75))
    air_match_fraction = float(np.mean(air_abs <= uniform_match_tolerance))
    material_match_fraction = float(np.mean(material_abs <= uniform_match_tolerance))
    repeat_tolerance = float(gates.get("repeat_max_abs_tolerance", 1.0e-6))
    circle_tolerance = float(gates.get("circle_mre_max", 0.005))
    failures = []
    for name, actual, threshold in (
        ("uniform_air_median_abs", air_median, uniform_tolerance),
        ("uniform_material_median_abs", material_median, uniform_tolerance),
        ("circle_repeat_max_abs", repeat_max, repeat_tolerance),
        ("circle_mre_bands_2_plus", circle_mre, circle_tolerance),
    ):
        if not np.isfinite(actual) or actual > threshold:
            failures.append(f"{name}={actual} exceeds {threshold}")
    for name, actual in (
        ("uniform_air_match_fraction", air_match_fraction),
        ("uniform_material_match_fraction", material_match_fraction),
    ):
        if not np.isfinite(actual) or actual < uniform_match_fraction_min:
            failures.append(f"{name}={actual} is below {uniform_match_fraction_min}")
    native_cluster = str(solver_payload.get("environment")) == "native_linux_conda_meep_env"
    if failures:
        status = "blocked"
    elif skip_environment_check:
        status = "solver_preflight_unaccepted"
    elif native_cluster:
        status = "cluster_solver_preflight_pass_pending_wsl_acceptance"
    else:
        status = "solver_preflight_pass"
    metrics = {
        "status": status,
        "uniform_air_max_abs": air_max,
        "uniform_material_max_abs": material_max,
        "uniform_air_median_abs": air_median,
        "uniform_material_median_abs": material_median,
        "uniform_air_match_fraction": air_match_fraction,
        "uniform_material_match_fraction": material_match_fraction,
        "circle_repeat_max_abs": repeat_max,
        "circle_mre_bands_2_plus": circle_mre,
        "uniform_median_abs_tolerance": uniform_tolerance,
        "uniform_match_abs_tolerance": uniform_match_tolerance,
        "uniform_match_fraction_min": uniform_match_fraction_min,
        "repeat_max_abs_tolerance": repeat_tolerance,
        "circle_mre_max": circle_tolerance,
        "k_point_count": int(len(k_points)),
        "case_times_s": timings,
        "mpb_generation_time_s": float(sum(timings.values())),
        "wall_time_total_s": float(time.perf_counter() - started),
        "validation_boundary": execution_profile["validation_boundary"],
        "execution_profile": execution_profile,
    }
    np.savez_compressed(out / "solver_preflight_bands.npz", k_points=k_points, **cases)
    write_json(out / "solver_preflight_cases.json", {"k_points": k_points.tolist(), "metrics": metrics, "failures": failures})
    _write_history(out / "history.csv", [{"stage": "solver_preflight", "status": status, "reason": "; ".join(failures)}])
    write_metrics(out, metrics)
    next_gate = (
        "full_cell_graded_and_binary_max4"
        if status == "solver_preflight_pass"
        else "cluster_full_cell_binary_max4_then_wsl_crosscheck"
        if status == "cluster_solver_preflight_pass_pending_wsl_acceptance"
        else "rerun_without_skip_environment_check"
        if status == "solver_preflight_unaccepted"
        else "repair_material_function_encoding"
    )
    write_results(out, {"status": status, "failures": failures, "next_gate": next_gate})
    write_run_meta(
        out,
        {
            "status": status,
            "validation_boundary": execution_profile["validation_boundary"] if not skip_environment_check else "unaccepted_environment_check_skipped",
            "config_path": _relative(config_path),
            "execution_profile": execution_profile,
        },
    )
    logger.info("Full-cell solver preflight completed: %s", status)
    return out


def write_combined_full_cell_manifest(
    path: str | Path,
    *,
    graded_manifest: str | Path,
    binary_manifest: str | Path,
    convergence_evidence: str | Path | None = None,
    status: str,
) -> Path:
    """Link accepted lanes without merging arrays or physical semantics."""

    out = _repo_path(path)
    entries = []
    for lane, value in (("graded_medium", graded_manifest), ("binary_air_holes", binary_manifest)):
        manifest_path = _repo_path(value)
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        entries.append({"lane": lane, "manifest": _relative(manifest_path), "manifest_sha256": _file_hash(manifest_path), "dataset": payload.get("dataset_path"), "dataset_sha256": payload.get("dataset_sha256"), "status": payload.get("status")})
    convergence = None
    if convergence_evidence is not None:
        convergence_path = _repo_path(convergence_evidence)
        convergence_payload = json.loads(convergence_path.read_text(encoding="utf-8"))
        convergence = {
            "path": _relative(convergence_path),
            "sha256": _file_hash(convergence_path),
            "status": convergence_payload.get("status"),
        }
    if str(status) == "pilot_pass":
        wrong_lane_statuses = [entry for entry in entries if entry["status"] != "pilot_generated_pending_convergence"]
        if wrong_lane_statuses:
            raise ValueError("pilot_pass requires both lane manifests to be complete real-MPB pilots")
        if convergence is None or convergence["status"] != "resolution_convergence_pass":
            raise ValueError("pilot_pass requires resolution_convergence_pass evidence")
    combined = {
        "schema_version": "phc2d.full_cell_collection.v1",
        "status": str(status),
        "merge_arrays": False,
        "datasets": entries,
        "resolution_convergence": convergence,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(combined, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


__all__ = [
    "load_full_cell_generation_config",
    "prepare_full_cell_sample_plan",
    "run_full_cell_generation",
    "run_full_cell_solver_preflight",
    "write_combined_full_cell_manifest",
]
