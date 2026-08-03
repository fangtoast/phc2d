"""Sample-level parallel worker helpers for MPB solves.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import get_context
from queue import Empty
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from phc2d.solvers.mpb import MpbBandSolverConfig, solve_bands_for_rho_mpb


SolverFn = Callable[..., tuple[np.ndarray, dict[str, Any]]]


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_array(array: np.ndarray) -> str:
    """Hash an array with shape and dtype metadata."""

    arr = np.ascontiguousarray(np.asarray(array))
    header = json.dumps({"shape": list(arr.shape), "dtype": str(arr.dtype)}, sort_keys=True).encode("utf-8")
    return _sha256_bytes(header + b"\0" + arr.tobytes())


@dataclass(frozen=True)
class MPBParallelConfig:
    """Config for sample-level MPB worker execution."""

    workers: int | str = 16
    min_workers: int = 12
    max_workers: int = 20
    worker_backend: str = "processpool"
    fail_fast: bool = True
    suppress_solver_output: bool = True
    worker_thread_env: dict[str, str] = field(
        default_factory=lambda: {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MPBParallelConfig":
        payload = dict(value or {})
        env = payload.get("worker_thread_env")
        return cls(
            workers=payload.get("workers", 16),
            min_workers=int(payload.get("min_workers", 12)),
            max_workers=int(payload.get("max_workers", 20)),
            worker_backend=str(payload.get("worker_backend", "processpool")),
            fail_fast=bool(payload.get("fail_fast", True)),
            suppress_solver_output=bool(payload.get("suppress_solver_output", True)),
            worker_thread_env={str(key): str(item) for key, item in dict(env or cls().worker_thread_env).items()},
        )

    def resolved_workers(self, sample_count: int, *, cpu_count: int | None = None) -> int:
        """Resolve the worker count for a concrete job list."""

        n_samples = max(0, int(sample_count))
        if n_samples == 0:
            return 0
        max_workers = max(1, int(self.max_workers))
        cpu = max(1, int(cpu_count or os.cpu_count() or 1))
        requested = self.workers
        if isinstance(requested, str) and requested.lower() == "auto":
            return max(1, min(16, max_workers, n_samples, cpu))
        return max(1, min(int(requested), max_workers, n_samples))

    def as_metadata(self, sample_count: int, *, cpu_count: int | None = None) -> dict[str, Any]:
        return {
            "workers_requested": self.workers,
            "workers_resolved": self.resolved_workers(sample_count, cpu_count=cpu_count),
            "min_workers": int(self.min_workers),
            "max_workers": int(self.max_workers),
            "worker_backend": self.worker_backend,
            "fail_fast": bool(self.fail_fast),
            "suppress_solver_output": bool(self.suppress_solver_output),
            "worker_thread_env": dict(self.worker_thread_env),
        }


@dataclass(frozen=True)
class MPBWorkerJob:
    """One independent MPB solve job."""

    sample_id: str
    sample_index: int
    split: str
    rho: np.ndarray
    k_points: Sequence[Sequence[float]]
    solver_config: MpbBandSolverConfig
    rho_hash: str = ""
    geometry_hash: str = ""
    stored: np.ndarray | None = None
    solver_input: Any | None = None


@dataclass
class MPBWorkerResult:
    """Result of one MPB worker job."""

    sample_id: str
    sample_index: int
    split: str
    status: str
    worker_id: str
    pid: int
    solve_time_s: float
    rho_hash: str
    geometry_hash: str
    Wh_hash: str = ""
    error: str = ""
    bands: np.ndarray | None = None
    solve_metadata: dict[str, Any] = field(default_factory=dict)
    resumed: bool = False

    def shard_payload(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "sample_index": self.sample_index,
            "split": self.split,
            "status": self.status,
            "worker_id": self.worker_id,
            "pid": self.pid,
            "solve_time_s": self.solve_time_s,
            "rho_hash": self.rho_hash,
            "geometry_hash": self.geometry_hash,
            "Wh_hash": self.Wh_hash,
            "error": self.error,
            "solve_metadata": self.solve_metadata,
            "resumed": self.resumed,
        }


def _result_from_payload(payload: Mapping[str, Any], bands: np.ndarray | None, *, resumed: bool) -> MPBWorkerResult:
    return MPBWorkerResult(
        sample_id=str(payload.get("sample_id", "")),
        sample_index=int(payload.get("sample_index", -1)),
        split=str(payload.get("split", "")),
        status=str(payload.get("status", "")),
        worker_id=str(payload.get("worker_id", "")),
        pid=int(payload.get("pid", 0)),
        solve_time_s=float(payload.get("solve_time_s", 0.0)),
        rho_hash=str(payload.get("rho_hash", "")),
        geometry_hash=str(payload.get("geometry_hash", "")),
        Wh_hash=str(payload.get("Wh_hash", "")),
        error=str(payload.get("error", "")),
        bands=bands,
        solve_metadata=dict(payload.get("solve_metadata") or {}),
        resumed=resumed,
    )


@contextmanager
def _silence_process_output(enabled: bool):
    """Temporarily redirect process-level stdout/stderr to os.devnull."""

    if not enabled:
        yield
        return
    saved: list[tuple[int, int]] = []
    devnull_fd: int | None = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        for fd in (1, 2):
            try:
                saved_fd = os.dup(fd)
                os.dup2(devnull_fd, fd)
                saved.append((fd, saved_fd))
            except OSError:
                continue
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            import ctypes

            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        for fd, saved_fd in reversed(saved):
            try:
                os.dup2(saved_fd, fd)
            finally:
                os.close(saved_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def _solve_worker(job: MPBWorkerJob, solver_fn: SolverFn, worker_env: Mapping[str, str], suppress_output: bool = False) -> MPBWorkerResult:
    old_env: dict[str, str | None] = {}
    for key, value in worker_env.items():
        old_env[str(key)] = os.environ.get(str(key))
        os.environ[str(key)] = str(value)
    started = time.perf_counter()
    try:
        with _silence_process_output(bool(suppress_output)):
            solver_input = job.rho if job.solver_input is None else job.solver_input
            bands, metadata = solver_fn(solver_input, k_points=job.k_points, config=job.solver_config)
        raw = np.asarray(bands)
        # Band-frequency workers use float32, while the field-eigen adapter
        # legitimately returns native-complex E/H tensors.  Never erase the
        # phase information at this shared shard boundary.
        bands = raw.astype(np.complex64 if np.iscomplexobj(raw) else np.float32, copy=False)
        status = "computed"
        error = ""
        wh_hash = hash_array(bands)
    except Exception as exc:  # pragma: no cover - exercised by tests with fake solver
        bands = None
        metadata = {}
        status = "solver_execution_failed"
        error = str(exc)
        wh_hash = ""
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return MPBWorkerResult(
        sample_id=job.sample_id,
        sample_index=int(job.sample_index),
        split=job.split,
        status=status,
        worker_id=str(os.getpid()),
        pid=int(os.getpid()),
        solve_time_s=float(time.perf_counter() - started),
        rho_hash=job.rho_hash or hash_array(job.rho),
        geometry_hash=job.geometry_hash,
        Wh_hash=wh_hash,
        error=error,
        bands=bands,
        solve_metadata=dict(metadata or {}),
    )


def _process_entry(job: MPBWorkerJob, solver_fn: SolverFn, worker_env: Mapping[str, str], suppress_output: bool, queue: Any) -> None:
    result = _solve_worker(job, solver_fn, worker_env, suppress_output=suppress_output)
    queue.put((result, result.bands))


def _load_completed_shard(job: MPBWorkerJob, shard_dir: Path) -> MPBWorkerResult | None:
    payload_path = shard_dir / f"{job.sample_id}.json"
    bands_path = shard_dir / f"{job.sample_id}.npy"
    if not payload_path.is_file() or not bands_path.is_file():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if str(payload.get("status")) != "computed":
        return None
    bands = np.load(bands_path)
    return _result_from_payload(payload, bands, resumed=True)


def _write_shard(result: MPBWorkerResult, shard_dir: Path) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_json(shard_dir / f"{result.sample_id}.json", result.shard_payload())
    if result.bands is not None:
        np.save(shard_dir / f"{result.sample_id}.npy", result.bands)


def _executor(config: MPBParallelConfig, workers: int):
    backend = str(config.worker_backend).lower()
    if backend in {"thread", "threads"}:
        return ThreadPoolExecutor(max_workers=workers)
    if backend not in {"spawn", "process", "processpool"}:
        raise ValueError("worker_backend must be subprocess, isolated_subprocess, spawn, process, processpool, or thread")
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))


def _run_pool_jobs(
    jobs: Sequence[MPBWorkerJob],
    *,
    config: MPBParallelConfig,
    workers: int,
    shard_root: Path,
    solver_fn: SolverFn,
) -> list[MPBWorkerResult]:
    pending_jobs = list(jobs)
    max_in_flight = max(1, int(workers) * 2)
    results: list[MPBWorkerResult] = []
    with _executor(config, workers) as pool:
        futures: dict[Any, MPBWorkerJob] = {}

        def submit_next() -> None:
            suppress = bool(config.suppress_solver_output) and str(config.worker_backend).lower() not in {"thread", "threads"}
            while pending_jobs and len(futures) < max_in_flight:
                job = pending_jobs.pop(0)
                futures[pool.submit(_solve_worker, job, solver_fn, config.worker_thread_env, suppress)] = job

        submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive around worker crashes
                    result = MPBWorkerResult(
                        sample_id=job.sample_id,
                        sample_index=job.sample_index,
                        split=job.split,
                        status="solver_execution_failed",
                        worker_id="0",
                        pid=0,
                        solve_time_s=0.0,
                        rho_hash=job.rho_hash or hash_array(job.rho),
                        geometry_hash=job.geometry_hash,
                        error=str(exc),
                    )
                _write_shard(result, shard_root)
                results.append(result)
                if result.status != "computed" and config.fail_fast:
                    for other in futures:
                        other.cancel()
                    futures.clear()
                    pending_jobs.clear()
                    break
            submit_next()
    return results


def _run_fresh_subprocess_jobs(
    jobs: Sequence[MPBWorkerJob],
    *,
    config: MPBParallelConfig,
    workers: int,
    shard_root: Path,
    solver_fn: SolverFn,
) -> list[MPBWorkerResult]:
    ctx = get_context("spawn")
    queue = ctx.Queue()
    pending = list(jobs)
    active: dict[str, tuple[Any, MPBWorkerJob]] = {}
    results: list[MPBWorkerResult] = []

    def launch_next() -> None:
        while pending and len(active) < workers:
            job = pending.pop(0)
            proc = ctx.Process(
                target=_process_entry,
                args=(job, solver_fn, config.worker_thread_env, bool(config.suppress_solver_output), queue),
            )
            proc.start()
            active[job.sample_id] = (proc, job)

    launch_next()
    while active:
        try:
            result, bands = queue.get(timeout=0.25)
            result.bands = bands
            _write_shard(result, shard_root)
            results.append(result)
            proc_job = active.pop(result.sample_id, None)
            if proc_job is not None:
                proc_job[0].join(timeout=1.0)
            if result.status != "computed" and config.fail_fast:
                for proc, _ in active.values():
                    proc.terminate()
                for proc, _ in active.values():
                    proc.join(timeout=1.0)
                active.clear()
                break
            launch_next()
        except Empty:
            exited: list[str] = []
            for sample_id, (proc, job) in active.items():
                if proc.is_alive():
                    continue
                proc.join(timeout=1.0)
                if proc.exitcode not in (0, None):
                    result = MPBWorkerResult(
                        sample_id=job.sample_id,
                        sample_index=job.sample_index,
                        split=job.split,
                        status="solver_execution_failed",
                        worker_id=str(proc.pid or 0),
                        pid=int(proc.pid or 0),
                        solve_time_s=0.0,
                        rho_hash=job.rho_hash or hash_array(job.rho),
                        geometry_hash=job.geometry_hash,
                        error=f"worker process exited with code {proc.exitcode}",
                    )
                    _write_shard(result, shard_root)
                    results.append(result)
                    exited.append(sample_id)
                    if config.fail_fast:
                        for other_proc, _ in active.values():
                            if other_proc is not proc:
                                other_proc.terminate()
                        for other_proc, _ in active.values():
                            other_proc.join(timeout=1.0)
                        active.clear()
                        break
                else:
                    exited.append(sample_id)
            for sample_id in exited:
                active.pop(sample_id, None)
            launch_next()
    return results


def run_mpb_worker_jobs(
    jobs: Sequence[MPBWorkerJob],
    *,
    config: MPBParallelConfig,
    shard_dir: str | Path,
    solver_fn: SolverFn = solve_bands_for_rho_mpb,
) -> tuple[list[MPBWorkerResult], dict[str, Any]]:
    """Run independent MPB jobs with resume-aware sample shards."""

    started = time.perf_counter()
    shard_root = Path(shard_dir)
    shard_root.mkdir(parents=True, exist_ok=True)
    ordered_jobs = sorted(list(jobs), key=lambda job: int(job.sample_index))
    workers = config.resolved_workers(len(ordered_jobs))
    results: list[MPBWorkerResult] = []
    pending: list[MPBWorkerJob] = []
    for job in ordered_jobs:
        resumed = _load_completed_shard(job, shard_root)
        if resumed is None:
            pending.append(job)
        else:
            results.append(resumed)

    if pending and workers == 1:
        suppress = bool(config.suppress_solver_output) and str(config.worker_backend).lower() not in {"thread", "threads"}
        for job in pending:
            result = _solve_worker(job, solver_fn, config.worker_thread_env, suppress_output=suppress)
            _write_shard(result, shard_root)
            results.append(result)
            if result.status != "computed" and config.fail_fast:
                break
    elif pending:
        backend = str(config.worker_backend).lower()
        if backend in {"subprocess", "isolated_subprocess"}:
            results.extend(
                _run_fresh_subprocess_jobs(
                    pending,
                    config=config,
                    workers=workers,
                    shard_root=shard_root,
                    solver_fn=solver_fn,
                )
            )
        else:
            results.extend(
                _run_pool_jobs(
                    pending,
                    config=config,
                    workers=workers,
                    shard_root=shard_root,
                    solver_fn=solver_fn,
                )
            )

    results = sorted(results, key=lambda item: int(item.sample_index))
    completed = [result for result in results if result.status == "computed"]
    failed = [result for result in results if result.status != "computed"]
    worker_pids = sorted({int(result.pid) for result in results if int(result.pid) > 0})
    solve_time = float(sum(float(result.solve_time_s) for result in results if result.status == "computed" and not result.resumed))
    wall_time = float(time.perf_counter() - started)
    summary = {
        **config.as_metadata(len(ordered_jobs)),
        "sample_count": int(len(ordered_jobs)),
        "completed_count": int(len(completed)),
        "failed_count": int(len(failed)),
        "resumed_count": int(sum(1 for result in results if result.resumed)),
        "worker_pids": worker_pids,
        "worker_solve_time_s": solve_time,
        "worker_wall_time_s": wall_time,
        "worker_throughput_samples_per_s": len(completed) / max(wall_time, 1.0e-12),
    }
    _write_json(shard_root.parent / "worker_summary.json", summary)
    return results, summary


__all__ = [
    "MPBParallelConfig",
    "MPBWorkerJob",
    "MPBWorkerResult",
    "hash_array",
    "run_mpb_worker_jobs",
]
