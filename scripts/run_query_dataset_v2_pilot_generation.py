"""Generate a provenance-complete V2 physical-k query pilot dataset.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.forward_dataset import ForwardDatasetBundle, save_forward_dataset
from phc2d.data.k_path import (
    KPathMetadata,
    build_square_gxm_reduced_spec,
    hash_k_path_points,
    hash_k_path_spec,
    save_k_path_spec,
)
from phc2d.data.lift import GridShape, lift_rho_to_grid, wedge_indices
from phc2d.generators import build_structure_generator, collect_structure_metadata
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_metrics, write_resolved_config, write_results, write_run_meta
from phc2d.solvers import MPBParallelConfig, MPBWorkerJob, MpbBandSolverConfig, run_mpb_worker_jobs, solve_bands_for_rho_mpb


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SOLVER_PROVENANCE_FIELDS = (
    "solver_backend",
    "execution_boundary",
    "lattice",
    "reciprocal_basis",
    "epsilon_air",
    "epsilon_material",
    "geometry_lifting_convention",
    "grid_shape",
    "n_wedge",
    "resolution",
    "polarization",
    "num_bands",
    "band_guard_count",
    "band_order_policy",
    "eigensolver_tolerance",
    "k_points",
    "endpoint_policy",
    "frequency_normalization",
    "generation_command",
    "source_code_commit",
    "config_hash",
    "run_id",
)


SolverFn = Callable[..., tuple[np.ndarray, dict[str, Any]]]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _jsonable(row.get(name, "")) for name in fieldnames})
    return path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    header = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}, sort_keys=True).encode("utf-8")
    return _sha256_bytes(header + b"\0" + arr.tobytes())


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(candidate)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(out.strip())


def load_query_pilot_config(path: str | Path) -> dict[str, Any]:
    """Load a Query Dataset V2 pilot YAML config."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Config must be a mapping: {path}")
    return dict(payload)


def derive_grid_shape(geometry_cfg: Mapping[str, Any]) -> GridShape:
    """Resolve the square grid shape from config."""

    raw = geometry_cfg.get("grid_shape")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError("geometry.grid_shape must be a two-item sequence resolved from config")
    rows, cols = int(raw[0]), int(raw[1])
    if rows <= 0 or rows != cols or rows % 2:
        raise ValueError(f"geometry.grid_shape must be a positive even square grid, got {raw}")
    return (rows, cols)


def derive_n_wedge(grid_shape: GridShape) -> int:
    """Resolve p4m wedge length from grid shape."""

    return int(len(wedge_indices(grid_shape)))


def _resolve_n_wedge(geometry_cfg: Mapping[str, Any], grid_shape: GridShape) -> int:
    derived = derive_n_wedge(grid_shape)
    raw = geometry_cfg.get("n_wedge", "from_grid_shape")
    if str(raw) == "from_grid_shape":
        return derived
    value = int(raw)
    if value != derived:
        raise ValueError(f"geometry.n_wedge={value} does not match grid_shape-derived value {derived}")
    return value


def _resolve_split_counts(dataset_cfg: Mapping[str, Any]) -> dict[str, int]:
    split = dict(dataset_cfg.get("split") or {})
    counts = {name: int(split.get(name, 0)) for name in ("train", "val", "test")}
    sample_count = int(dataset_cfg["sample_count"])
    if sum(counts.values()) != sample_count:
        raise ValueError(f"dataset.split counts {counts} do not sum to dataset.sample_count={sample_count}")
    return counts


def _effective_split_counts(full_counts: Mapping[str, int], effective_count: int) -> dict[str, int]:
    total = int(sum(int(v) for v in full_counts.values()))
    n = int(effective_count)
    if n == total:
        return {name: int(full_counts[name]) for name in ("train", "val", "test")}
    if n <= 0:
        raise ValueError("effective sample count must be positive")
    exact = {name: n * int(full_counts[name]) / max(total, 1) for name in ("train", "val", "test")}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    remaining = n - sum(counts.values())
    for name, _ in sorted(exact.items(), key=lambda item: item[1] - np.floor(item[1]), reverse=True):
        if remaining <= 0:
            break
        counts[name] += 1
        remaining -= 1
    return counts


def _split_labels(counts: Mapping[str, int]) -> list[str]:
    labels: list[str] = []
    for name in ("train", "val", "test"):
        labels.extend([name] * int(counts.get(name, 0)))
    return labels


def _split_indices(labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(labels, dtype=object)
    return (
        np.flatnonzero(arr == "train").astype(np.int64),
        np.flatnonzero(arr == "val").astype(np.int64),
        np.flatnonzero(arr == "test").astype(np.int64),
    )


def _generation_command(config_path: Path, max_samples: int | None) -> str:
    parts = [
        "PYTHONPATH=src",
        "python",
        "scripts/run_query_dataset_v2_pilot_generation.py",
        "--config",
        _display_path(config_path),
    ]
    if max_samples is not None:
        parts.extend(["--max-samples", str(int(max_samples))])
    return " ".join(parts)


def _solver_parallel_config(
    solver_cfg: Mapping[str, Any],
    *,
    workers: int | str | None = None,
    min_workers: int | None = None,
    max_workers: int | None = None,
    worker_backend: str | None = None,
    force_serial_default: bool = False,
) -> MPBParallelConfig:
    payload = dict(solver_cfg)
    if force_serial_default and "workers" not in payload and workers is None:
        payload["workers"] = 1
        payload["min_workers"] = 1
        payload["max_workers"] = 1
    if workers is not None:
        payload["workers"] = workers
    if min_workers is not None:
        payload["min_workers"] = min_workers
    if max_workers is not None:
        payload["max_workers"] = max_workers
    if worker_backend is not None:
        payload["worker_backend"] = worker_backend
    return MPBParallelConfig.from_mapping(payload)


def _execution_boundary_problems() -> list[str]:
    problems: list[str] = []
    release = platform.release().lower()
    proc_version = ""
    proc_path = Path("/proc/version")
    if proc_path.is_file():
        proc_version = proc_path.read_text(encoding="utf-8", errors="ignore").lower()
    if os.environ.get("WSL_DISTRO_NAME") != "Ubuntu" and "microsoft" not in release and "microsoft" not in proc_version:
        problems.append("not running inside WSL Ubuntu")
    if os.environ.get("CONDA_DEFAULT_ENV") != "meep_env":
        problems.append("conda environment is not meep_env")
    return problems


def _validate_contract(cfg: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    solver = dict(cfg.get("solver") or {})
    geometry = dict(cfg.get("geometry") or {})
    dataset = dict(cfg.get("dataset") or {})
    output = dict(cfg.get("output") or {})
    k_path = dict(cfg.get("k_path") or {})
    if str(solver.get("backend", "")).lower() != "mpb":
        problems.append("solver.backend must be mpb")
    if bool(solver.get("use_proxy", True)):
        problems.append("solver.use_proxy must be false")
    if str(solver.get("environment", "")) != "wsl_meep_env":
        problems.append("solver.environment must be wsl_meep_env")
    if solver.get("eigensolver_tolerance") in (None, ""):
        problems.append("solver.eigensolver_tolerance is required")
    if int(solver.get("band_guard_count", 0)) < 0:
        problems.append("solver.band_guard_count must be non-negative")
    if str(solver.get("band_order_policy", "frequency_ascending")) != "frequency_ascending":
        problems.append("solver.band_order_policy must be frequency_ascending for canonical V2 retained-band output")
    for key in ("epsilon_air", "epsilon_material", "resolution", "polarization", "num_bands"):
        if solver.get(key) in (None, ""):
            problems.append(f"solver.{key} is required")
    for key in ("output_path", "sample_count", "split", "sample_id_policy"):
        if dataset.get(key) in (None, ""):
            problems.append(f"dataset.{key} is required")
    if str(dataset.get("sample_id_policy", "")) != "stable_generated_id":
        problems.append("dataset.sample_id_policy must be stable_generated_id")
    if geometry.get("grid_shape") in (None, ""):
        problems.append("geometry.grid_shape must be resolved from config")
    if str(geometry.get("lifting", "")) != "p4m_wedge_binary_pixels_as_mpb_blocks_air_background":
        problems.append("geometry.lifting must record the p4m MPB block convention")
    if k_path.get("n_total") in (None, ""):
        problems.append("k_path.n_total is required")
    for key in ("write_manifest", "write_k_path_sidecar", "write_solver_provenance", "write_metadata_jsonl"):
        if output.get(key) is not True:
            problems.append(f"output.{key} must be true")
    try:
        grid_shape = derive_grid_shape(geometry)
        _resolve_n_wedge(geometry, grid_shape)
        _resolve_split_counts(dataset)
    except Exception as exc:
        problems.append(str(exc))
    return problems


def _build_k_path_spec(
    cfg: Mapping[str, Any],
    *,
    config_path: Path,
    config_hash: str,
    run_id: str,
    status: str,
    git_commit: str,
) -> Any:
    k_cfg = dict(cfg.get("k_path") or {})
    metadata = KPathMetadata(
        source_type=str(k_cfg.get("source", "generated_from_KPathSpec")),
        source_path=_display_path(config_path),
        source_commit=git_commit,
        source_run_id=run_id,
        source_config_hash=config_hash,
        verification_status=status,
        verification_run_id=run_id,
        notes=(
            "Generated for Query Dataset V2 pilot; not a Formal Full-2A benchmark.",
            f"endpoint_policy={k_cfg.get('endpoint_policy', 'explicit')}",
        ),
    )
    return build_square_gxm_reduced_spec(
        n_total=int(k_cfg["n_total"]),
        lattice_constant=float(dict(cfg.get("solver") or {}).get("lattice_constant", 1.0)),
        k_path_id=str(k_cfg.get("k_path_id", "query_dataset_v2_pilot_square_gxm_mpb_reduced_v1")),
        metadata=metadata,
    )


def build_solver_provenance(
    cfg: Mapping[str, Any],
    *,
    grid_shape: GridShape,
    n_wedge: int,
    k_points: Sequence[Sequence[float]],
    config_hash: str,
    run_id: str,
    config_path: Path,
    generation_command: str,
    git_commit: str,
    status: str,
) -> dict[str, Any]:
    """Build machine-readable solver provenance for the generation run."""

    solver = dict(cfg.get("solver") or {})
    geometry = dict(cfg.get("geometry") or {})
    worker_cfg = MPBParallelConfig.from_mapping(solver)
    fields = {
        "solver_backend": {"value": "real_mpb", "status": "available"},
        "execution_boundary": {"value": "WSL Ubuntu + conda meep_env", "status": "available"},
        "lattice": {"value": solver.get("lattice"), "status": "available"},
        "reciprocal_basis": {"value": "square reciprocal basis; MPB reduced reciprocal coordinates", "status": "available"},
        "epsilon_air": {"value": solver.get("epsilon_air"), "status": "available"},
        "epsilon_material": {"value": solver.get("epsilon_material"), "status": "available"},
        "geometry_lifting_convention": {"value": geometry.get("lifting"), "status": "available"},
        "grid_shape": {"value": list(grid_shape), "source": "geometry.grid_shape", "status": "available"},
        "n_wedge": {"value": int(n_wedge), "source": "derived_from_grid_shape", "status": "available"},
        "resolution": {"value": solver.get("resolution"), "status": "available"},
        "polarization": {"value": solver.get("polarization"), "status": "available"},
        "num_bands": {"value": solver.get("num_bands"), "status": "available"},
        "band_guard_count": {"value": int(solver.get("band_guard_count", 0)), "status": "available"},
        "band_order_policy": {
            "value": str(solver.get("band_order_policy", "frequency_ascending")),
            "status": "available",
            "source": _display_path(config_path),
        },
        "solved_num_bands": {
            "value": int(solver.get("num_bands")) + int(solver.get("band_guard_count", 0)) if solver.get("num_bands") is not None else None,
            "status": "available" if solver.get("num_bands") is not None else "missing",
        },
        "eigensolver_tolerance": {
            "value": solver.get("eigensolver_tolerance"),
            "status": "available" if solver.get("eigensolver_tolerance") not in (None, "") else "missing",
            "source": _display_path(config_path),
        },
        "k_points": {
            "value": {
                "count": int(len(k_points)),
                "hash": hash_k_path_points(k_points),
                "coordinate_system": dict(cfg.get("k_path") or {}).get("coordinate_system"),
            },
            "status": "available",
        },
        "endpoint_policy": {"value": dict(cfg.get("k_path") or {}).get("endpoint_policy"), "status": "available"},
        "frequency_normalization": {"value": solver.get("frequency_normalization"), "status": "available"},
        "generation_command": {"value": generation_command, "status": "available"},
        "source_code_commit": {"value": git_commit, "dirty_worktree": _git_dirty(), "status": "available"},
        "config_hash": {"value": config_hash, "status": "available"},
        "run_id": {"value": run_id, "status": "available"},
    }
    missing = [
        name
        for name in REQUIRED_SOLVER_PROVENANCE_FIELDS
        if fields.get(name, {}).get("status") == "missing" or fields.get(name, {}).get("value") in (None, "")
    ]
    return {
        "schema_version": "phc2d.query_dataset_v2.solver_provenance.v1",
        "status": status,
        "solver_backend": "real_mpb",
        "use_proxy": bool(solver.get("use_proxy", True)),
        "required_fields": list(REQUIRED_SOLVER_PROVENANCE_FIELDS),
        "fields": fields,
        "missing_required_fields": missing,
        "solver_provenance_complete": not missing,
        "worker_policy": worker_cfg.as_metadata(int(dict(cfg.get("dataset") or {}).get("sample_count", 0))),
    }


def _dataset_output_path(cfg: Mapping[str, Any], run_dir: Path, *, max_samples: int | None) -> Path:
    configured = _repo_path(str(dict(cfg.get("dataset") or {})["output_path"]))
    if max_samples is None:
        return configured
    stem = configured.stem
    return run_dir / f"{stem}_smoke_{int(max_samples)}.npz"


def _metadata_paths(dataset_path: Path) -> tuple[Path, Path]:
    metadata_dir = dataset_path.parent / "metadata"
    return (
        metadata_dir / f"{dataset_path.stem}.kpath.v1.json",
        metadata_dir / f"{dataset_path.stem}.manifest.yaml",
    )


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    return _write_csv(path, rows, ["stage", "status", "reason"])


def _write_common_artifacts(
    run_dir: Path,
    *,
    resolved_yaml: str,
    solver_provenance: Mapping[str, Any],
    k_path_spec: Any,
    manifest: Mapping[str, Any],
    sample_metadata: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    per_sample_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    sample_plan: Sequence[Mapping[str, Any]] | None = None,
    worker_summary: Mapping[str, Any] | None = None,
) -> None:
    ensure_figures_dir(run_dir)
    write_resolved_config(run_dir, resolved_yaml)
    _write_json(run_dir / "solver_provenance.json", solver_provenance)
    save_k_path_spec(k_path_spec, run_dir / "k_path_sidecar.json")
    _write_yaml(run_dir / "dataset_manifest.yaml", manifest)
    _write_jsonl(run_dir / "sample_plan.jsonl", sample_plan or [])
    if worker_summary is not None:
        _write_json(run_dir / "worker_summary.json", worker_summary)
    _write_jsonl(run_dir / "sample_metadata.jsonl", sample_metadata)
    _write_csv(
        run_dir / "selected_or_generated_samples.csv",
        selected_rows,
        [
            "sample_id",
            "split",
            "sample_index",
            "rho_hash",
            "geometry_hash",
            "Wh_hash",
            "solver_provenance_ref",
            "status",
        ],
    )
    _write_csv(
        run_dir / "per_sample_generation_metrics.csv",
        per_sample_rows,
        ["sample_id", "split", "sample_index", "mpb_solve_time_s", "rho_hash", "geometry_hash", "Wh_hash", "worker_id", "pid", "status"],
    )
    _write_csv(run_dir / "failure_cases.csv", failures, ["sample_id", "split", "sample_index", "status", "reason"])
    _write_history(run_dir / "history.csv", history_rows)


def _write_blocked_outputs(
    run_dir: Path,
    *,
    cfg: Mapping[str, Any],
    config_path: Path,
    resolved_yaml: str,
    config_hash: str,
    status: str,
    reason: str,
    max_samples: int | None,
) -> None:
    grid_shape = derive_grid_shape(dict(cfg.get("geometry") or {}))
    n_wedge = _resolve_n_wedge(dict(cfg.get("geometry") or {}), grid_shape)
    run_status = "blocked"
    git_commit = _git_commit()
    k_path_spec = _build_k_path_spec(
        cfg,
        config_path=config_path,
        config_hash=config_hash,
        run_id=run_dir.name,
        status=run_status,
        git_commit=git_commit,
    )
    solver_provenance = build_solver_provenance(
        cfg,
        grid_shape=grid_shape,
        n_wedge=n_wedge,
        k_points=k_path_spec.k_points,
        config_hash=config_hash,
        run_id=run_dir.name,
        config_path=config_path,
        generation_command=_generation_command(config_path, max_samples),
        git_commit=git_commit,
        status=run_status,
    )
    manifest = {
        "dataset_id": Path(str(dict(cfg.get("dataset") or {}).get("output_path", "query_dataset_v2_pilot"))).stem,
        "status": status,
        "reason": reason,
        "dataset_path": "",
        "k_path_sidecar": "k_path_sidecar.json",
        "solver_provenance": "solver_provenance.json",
        "provenance_status": status,
        "formal_full_2a_allowed": False,
    }
    _write_common_artifacts(
        run_dir,
        resolved_yaml=resolved_yaml,
        solver_provenance=solver_provenance,
        k_path_spec=k_path_spec,
        manifest=manifest,
        sample_metadata=[],
        selected_rows=[],
        per_sample_rows=[],
        failures=[{"sample_id": "", "split": "", "sample_index": "", "status": status, "reason": reason}],
        history_rows=[{"stage": "pre_generation", "status": status, "reason": reason}],
    )
    write_metrics(
        run_dir,
        {
            "physical_validation_decision": status,
            "real_mpb_generation_started": False,
            "real_mpb_generation_completed": False,
            "wall_time_total_s": 0.0,
            "sample_count": 0,
            "missing_required_fields": solver_provenance.get("missing_required_fields", []),
        },
    )
    write_results(
        run_dir,
        {
            "action": "query_dataset_v2_pilot_generation",
            "status": status,
            "reason": reason,
            "formal_full_2a_allowed": False,
            "dataset_written": False,
        },
    )
    write_run_meta(
        run_dir,
        {
            "script": "scripts/run_query_dataset_v2_pilot_generation.py",
            "config_path": _display_path(config_path),
            "validation_boundary": "wsl_meep_env_real_mpb_generation",
            "status": status,
        },
    )


def run_generation(
    config_path: str | Path,
    *,
    run_dir_override: str | Path | None = None,
    max_samples: int | None = None,
    workers: int | str | None = None,
    min_workers: int | None = None,
    max_workers: int | None = None,
    worker_backend: str | None = None,
    skip_environment_check: bool = False,
    solver_fn: SolverFn = solve_bands_for_rho_mpb,
) -> Path:
    """Generate the V2 pilot dataset or write a machine-readable blocked run."""

    wall_start = time.perf_counter()
    config_path = Path(config_path)
    cfg = load_query_pilot_config(config_path)
    experiment_id = str(cfg.get("experiment_id", "query_dataset_v2_pilot"))
    runs_root = Path(str(dict(cfg.get("output") or {}).get("runs_dir", "runs")))
    run_dir = make_run_dir(experiment_id, runs_dir=runs_root, run_dir=run_dir_override)
    logger = setup_logging(
        log_file=run_dir / "run.log",
        run_meta={
            "script": "scripts/run_query_dataset_v2_pilot_generation.py",
            "config_path": _display_path(config_path),
            "validation_boundary": "wsl_meep_env_real_mpb_generation",
        },
    )

    run_options = {
        "max_samples": max_samples,
        "status_policy": "pilot_smoke_when_max_samples_is_set",
        "workers_override": workers,
        "min_workers_override": min_workers,
        "max_workers_override": max_workers,
        "worker_backend_override": worker_backend,
    }
    resolved_cfg = dict(cfg)
    resolved_cfg["run_options"] = run_options
    resolved_yaml = yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True)
    config_hash = _sha256_text(resolved_yaml)
    git_commit = _git_commit()

    contract_problems = _validate_contract(cfg)
    if contract_problems:
        reason = "; ".join(contract_problems)
        _write_blocked_outputs(
            run_dir,
            cfg=cfg,
            config_path=config_path,
            resolved_yaml=resolved_yaml,
            config_hash=config_hash,
            status="blocked",
            reason=reason,
            max_samples=max_samples,
        )
        logger.error("Generation blocked: %s", reason)
        return run_dir

    if not skip_environment_check:
        boundary_problems = _execution_boundary_problems()
        if boundary_problems:
            reason = "; ".join(boundary_problems)
            _write_blocked_outputs(
                run_dir,
                cfg=cfg,
                config_path=config_path,
                resolved_yaml=resolved_yaml,
                config_hash=config_hash,
                status="blocked",
                reason=reason,
                max_samples=max_samples,
            )
            logger.error("Generation blocked by execution boundary: %s", reason)
            return run_dir

    dataset_cfg = dict(cfg.get("dataset") or {})
    geometry_cfg = dict(cfg.get("geometry") or {})
    solver_cfg = dict(cfg.get("solver") or {})
    grid_shape = derive_grid_shape(geometry_cfg)
    n_wedge = _resolve_n_wedge(geometry_cfg, grid_shape)
    sample_count = int(dataset_cfg["sample_count"])
    effective_count = min(sample_count, int(max_samples)) if max_samples is not None else sample_count
    full_split_counts = _resolve_split_counts(dataset_cfg)
    split_counts = _effective_split_counts(full_split_counts, effective_count)
    labels = _split_labels(split_counts)
    idx_train, idx_val, idx_test = _split_indices(labels)
    run_status = "pilot_smoke" if max_samples is not None and effective_count < sample_count else "pilot_generated"

    k_path_spec = _build_k_path_spec(
        cfg,
        config_path=config_path,
        config_hash=config_hash,
        run_id=run_dir.name,
        status=run_status,
        git_commit=git_commit,
    )
    solver_provenance = build_solver_provenance(
        cfg,
        grid_shape=grid_shape,
        n_wedge=n_wedge,
        k_points=k_path_spec.k_points,
        config_hash=config_hash,
        run_id=run_dir.name,
        config_path=config_path,
        generation_command=_generation_command(config_path, max_samples),
        git_commit=git_commit,
        status=run_status,
    )
    if not solver_provenance["solver_provenance_complete"]:
        reason = "missing required solver provenance: " + ", ".join(solver_provenance["missing_required_fields"])
        _write_blocked_outputs(
            run_dir,
            cfg=cfg,
            config_path=config_path,
            resolved_yaml=resolved_yaml,
            config_hash=config_hash,
            status="blocked",
            reason=reason,
            max_samples=max_samples,
        )
        logger.error(reason)
        return run_dir

    dataset_output = _dataset_output_path(cfg, run_dir, max_samples=max_samples)
    data_sidecar_path, data_manifest_path = _metadata_paths(dataset_output)
    solver = MpbBandSolverConfig(
        num_bands=int(solver_cfg["num_bands"]),
        resolution=int(solver_cfg["resolution"]),
        polarization=str(solver_cfg["polarization"]),
        epsilon_air=float(solver_cfg["epsilon_air"]),
        epsilon_material=float(solver_cfg["epsilon_material"]),
        lattice_constant=float(solver_cfg.get("lattice_constant", 1.0)),
        eigensolver_tolerance=float(solver_cfg["eigensolver_tolerance"]),
        grid_shape=grid_shape,
        band_guard_count=int(solver_cfg.get("band_guard_count", 0)),
        band_order_policy=str(solver_cfg.get("band_order_policy", "frequency_ascending")),
    )
    generator_cfg = dict(dataset_cfg)
    generator_cfg["n_wedge"] = n_wedge
    generator_cfg["grid_shape"] = list(grid_shape)
    generator = build_structure_generator(generator_cfg)
    rng = np.random.default_rng(int(dataset_cfg.get("seed", 0)))
    parallel_cfg = _solver_parallel_config(
        solver_cfg,
        workers=workers,
        min_workers=min_workers,
        max_workers=max_workers,
        worker_backend=worker_backend,
        force_serial_default=solver_fn is not solve_bands_for_rho_mpb,
    )

    rho_rows: list[np.ndarray] = []
    wh_rows: list[np.ndarray] = []
    sample_metadata: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = [{"stage": "provenance_check", "status": "passed", "reason": ""}]
    sample_plan: list[dict[str, Any]] = []
    jobs: list[MPBWorkerJob] = []

    for sample_index in range(effective_count):
        split = labels[sample_index]
        sample_id = f"{experiment_id}_{sample_index:06d}"
        sample = generator.sample(rng=rng, dataset_cfg=generator_cfg)
        rho = np.asarray(sample.rho, dtype=np.float32).reshape(-1)
        if int(rho.shape[0]) != n_wedge:
            raise ValueError(f"Generated rho length {rho.shape[0]} does not match config-derived n_wedge={n_wedge}")
        binary_grid = np.asarray(sample.binary_grid, dtype=np.float32)
        if tuple(int(v) for v in binary_grid.shape) != grid_shape:
            binary_grid = lift_rho_to_grid(rho, grid_shape=grid_shape).astype(np.float32)

        rho_hash = _hash_array(rho)
        geometry_hash = _hash_array((binary_grid > 0.5).astype(np.uint8))
        structure_meta = collect_structure_metadata(
            binary_grid=binary_grid,
            seed=int(dataset_cfg.get("seed", 0)) + sample_index,
            sample_size=int(grid_shape[0]),
            generator_name=str(getattr(generator, "name", "unknown")),
            symmetry=str(sample.metadata.get("symmetry", "")),
            postprocess_steps=list(sample.postprocess_steps),
        )
        row = {
            "sample_id": sample_id,
            "split": split,
            "sample_index": sample_index,
            "rho_hash": rho_hash,
            "geometry_hash": geometry_hash,
            "solver_provenance_ref": "solver_provenance.json",
            "k_path_sidecar_ref": "k_path_sidecar.json",
            "generation_status": run_status,
            "structure_metadata": {**structure_meta, **dict(sample.metadata)},
        }
        sample_plan.append(row)
        jobs.append(
            MPBWorkerJob(
                sample_id=sample_id,
                sample_index=sample_index,
                split=split,
                rho=rho,
                k_points=k_path_spec.k_points,
                solver_config=solver,
                rho_hash=rho_hash,
                geometry_hash=geometry_hash,
            )
        )
        rho_rows.append(rho)
    _write_jsonl(run_dir / "sample_plan.jsonl", sample_plan)

    worker_results, worker_summary = run_mpb_worker_jobs(
        jobs,
        config=parallel_cfg,
        shard_dir=run_dir / "worker_shards",
        solver_fn=solver_fn,
    )
    _write_json(run_dir / "worker_summary.json", worker_summary)

    for result in worker_results:
        plan_row = sample_plan[int(result.sample_index)]
        if result.status != "computed" or result.bands is None:
            failures.append(
                {
                    "sample_id": result.sample_id,
                    "split": result.split,
                    "sample_index": result.sample_index,
                    "status": result.status,
                    "reason": result.error,
                }
            )
            continue
        wh = np.asarray(result.bands, dtype=np.float32)
        expected_shape = (int(k_path_spec.Nk), int(solver.num_bands))
        if tuple(int(v) for v in wh.shape) != expected_shape:
            raise RuntimeError(f"Solver returned Wh shape {wh.shape}, expected {expected_shape}")
        row = {
            **plan_row,
            "Wh_hash": result.Wh_hash,
            "mpb_solve_metadata": result.solve_metadata,
        }
        sample_metadata.append(row)
        selected_rows.append({**row, "status": "generated"})
        per_sample_rows.append(
            {
                "sample_id": result.sample_id,
                "split": result.split,
                "sample_index": result.sample_index,
                "mpb_solve_time_s": result.solve_time_s,
                "rho_hash": result.rho_hash,
                "geometry_hash": result.geometry_hash,
                "Wh_hash": result.Wh_hash,
                "worker_id": result.worker_id,
                "pid": result.pid,
                "status": "generated",
            }
        )
        wh_rows.append(wh)
        logger.info("Generated sample %s (%s/%s)", result.sample_id, int(result.sample_index) + 1, effective_count)

    if failures:
        status = "solver_execution_failed"
        reason = failures[0]["reason"]
        manifest = {
            "dataset_id": dataset_output.stem,
            "status": status,
            "reason": reason,
            "dataset_path": "",
            "k_path_sidecar": "k_path_sidecar.json",
            "solver_provenance": "solver_provenance.json",
            "sample_metadata": "sample_metadata.jsonl",
            "provenance_status": status,
            "formal_full_2a_allowed": False,
        }
        history_rows.append({"stage": "mpb_generation", "status": status, "reason": reason})
        _write_common_artifacts(
            run_dir,
            resolved_yaml=resolved_yaml,
            solver_provenance=solver_provenance,
            k_path_spec=k_path_spec,
            manifest=manifest,
            sample_metadata=sample_metadata,
            selected_rows=selected_rows,
            per_sample_rows=per_sample_rows,
            failures=failures,
            history_rows=history_rows,
            sample_plan=sample_plan,
            worker_summary=worker_summary,
        )
        write_metrics(
            run_dir,
            {
                "physical_validation_decision": status,
                "real_mpb_generation_started": bool(worker_results),
                "real_mpb_generation_completed": False,
                "wall_time_total_s": float(time.perf_counter() - wall_start),
                "sample_count": int(len(rho_rows)),
                "requested_sample_count": sample_count,
                "effective_sample_count": effective_count,
                "mpb_generation_time_s": float(sum(float(row["mpb_solve_time_s"]) for row in per_sample_rows)),
                **worker_summary,
            },
        )
        write_results(
            run_dir,
            {
                "action": "query_dataset_v2_pilot_generation",
                "status": status,
                "reason": reason,
                "formal_full_2a_allowed": False,
                "dataset_written": False,
                "generated_sample_count": int(len(rho_rows)),
            },
        )
        write_run_meta(
            run_dir,
            {
                "script": "scripts/run_query_dataset_v2_pilot_generation.py",
                "config_path": _display_path(config_path),
                "validation_boundary": "wsl_meep_env_real_mpb_generation",
                "status": status,
            },
        )
        return run_dir

    rho_array = np.stack(rho_rows, axis=0).astype(np.float32)
    wh_array = np.stack(wh_rows, axis=0).astype(np.float32)
    bundle = ForwardDatasetBundle(
        rho=rho_array,
        Wh=wh_array,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        config_yaml=resolved_yaml,
        metadata=sample_metadata,
    )
    save_forward_dataset(dataset_output, bundle)
    save_k_path_spec(k_path_spec, data_sidecar_path)

    manifest = {
        "dataset_path": _display_path(dataset_output),
        "dataset_id": dataset_output.stem,
        "dataset_file_hash": _file_hash(dataset_output),
        "Wh_shape": list(wh_array.shape),
        "rho_shape": list(rho_array.shape),
        "split": {name: int(split_counts.get(name, 0)) for name in ("train", "val", "test")},
        "k_path_sidecar": data_sidecar_path.name,
        "k_path_hash": hash_k_path_spec(k_path_spec),
        "k_points_hash": hash_k_path_points(k_path_spec.k_points),
        "solver_provenance": _display_path(run_dir / "solver_provenance.json"),
        "config_hash": config_hash,
        "sample_id_policy": dataset_cfg.get("sample_id_policy"),
        "sample_metadata": _display_path(run_dir / "sample_metadata.jsonl"),
        "provenance_status": run_status,
        "physical_verification_status": run_status,
        "formal_full_2a_allowed": False,
        "association_note": "V2 pilot generated with real MPB provenance; not a Formal Full-2A benchmark.",
    }
    _write_yaml(data_manifest_path, manifest)
    history_rows.append({"stage": "mpb_generation", "status": run_status, "reason": ""})
    history_rows.append({"stage": "dataset_write", "status": "completed", "reason": ""})
    _write_common_artifacts(
        run_dir,
        resolved_yaml=resolved_yaml,
        solver_provenance=solver_provenance,
        k_path_spec=k_path_spec,
        manifest=manifest,
        sample_metadata=sample_metadata,
        selected_rows=selected_rows,
        per_sample_rows=per_sample_rows,
        failures=[],
        history_rows=history_rows,
        sample_plan=sample_plan,
        worker_summary=worker_summary,
    )
    wall_time = float(time.perf_counter() - wall_start)
    mpb_time = float(sum(float(row["mpb_solve_time_s"]) for row in per_sample_rows))
    write_metrics(
        run_dir,
        {
            "physical_validation_decision": run_status,
            "real_mpb_generation_started": True,
            "real_mpb_generation_completed": True,
            "wall_time_total_s": wall_time,
            "mpb_generation_time_s": mpb_time,
            "mpb_generation_time_per_sample_s": mpb_time / max(effective_count, 1),
            "throughput_samples_per_s": effective_count / max(wall_time, 1.0e-12),
            "sample_count": int(effective_count),
            "requested_sample_count": int(sample_count),
            "effective_sample_count": int(effective_count),
            "pilot_smoke": run_status == "pilot_smoke",
            **worker_summary,
        },
    )
    write_results(
        run_dir,
        {
            "action": "query_dataset_v2_pilot_generation",
            "status": run_status,
            "formal_full_2a_allowed": False,
            "dataset_written": True,
            "dataset_path": _display_path(dataset_output),
            "k_path_sidecar": _display_path(data_sidecar_path),
            "dataset_manifest": _display_path(data_manifest_path),
            "solver_provenance": _display_path(run_dir / "solver_provenance.json"),
            "generated_sample_count": int(effective_count),
            "old_50k_status": "unchanged_candidate_unverified",
        },
    )
    write_run_meta(
        run_dir,
        {
            "script": "scripts/run_query_dataset_v2_pilot_generation.py",
            "config_path": _display_path(config_path),
            "validation_boundary": "wsl_meep_env_real_mpb_generation",
            "status": run_status,
            "dataset_path": _display_path(dataset_output),
        },
    )
    logger.info("Query Dataset V2 pilot generation finished: %s", run_status)
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate a Query Dataset V2 physical-k MPB pilot dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Generate a pilot_smoke subset instead of the full configured dataset.")
    parser.add_argument("--workers", default=None, help="MPB worker count: integer or auto.")
    parser.add_argument("--min-workers", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--worker-backend", default=None)
    args = parser.parse_args(argv)
    workers: int | str | None = None
    if args.workers is not None:
        workers = str(args.workers)
        if workers.lower() != "auto":
            workers = int(workers)
    run_dir = run_generation(
        args.config,
        run_dir_override=args.run_dir,
        max_samples=args.max_samples,
        workers=workers,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        worker_backend=args.worker_backend,
    )
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
