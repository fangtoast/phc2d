"""Tests for sample-level MPB worker orchestration.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from phc2d.solvers import (
    MPBParallelConfig,
    MPBWorkerJob,
    MpbBandSolverConfig,
    run_mpb_worker_jobs,
    run_mpb_worker_partition,
)


def _solver_config() -> MpbBandSolverConfig:
    return MpbBandSolverConfig(
        num_bands=2,
        resolution=8,
        polarization="TE",
        epsilon_air=1.0,
        epsilon_material=8.9,
        eigensolver_tolerance=1.0e-6,
        grid_shape=(4, 4),
        band_guard_count=1,
    )


def _jobs(count: int) -> list[MPBWorkerJob]:
    return [
        MPBWorkerJob(
            sample_id=f"unit_{idx:06d}",
            sample_index=idx,
            split="train",
            rho=np.full(3, idx, dtype=np.float32),
            k_points=[[0.0, 0.0], [0.5, 0.0]],
            solver_config=_solver_config(),
            rho_hash=f"rho{idx}",
            geometry_hash=f"geom{idx}",
        )
        for idx in range(count)
    ]


def sleepy_fake_solver(rho, *, k_points, config):
    time.sleep(0.15)
    value = float(np.asarray(rho).mean())
    return np.full((len(k_points), int(config.num_bands)), value, dtype=np.float32), {"backend": "sleepy_fake"}


def failing_fake_solver(rho, *, k_points, config):
    if float(np.asarray(rho).mean()) == 1.0:
        raise RuntimeError("unit failure")
    return np.zeros((len(k_points), int(config.num_bands)), dtype=np.float32), {"backend": "failing_fake"}


def solver_input_fake_solver(value, *, k_points, config):
    if value != {"kind": "analytic"}:
        raise AssertionError("worker did not dispatch solver_input")
    return np.ones((len(k_points), int(config.num_bands)), dtype=np.float32), {"backend": "solver_input_fake"}


def test_auto_workers_clamp_to_default_max_cpu_and_sample_count() -> None:
    assert MPBParallelConfig().worker_backend == "processpool"
    assert MPBParallelConfig(workers="auto").resolved_workers(30, cpu_count=64) == 16
    assert MPBParallelConfig(workers="auto").resolved_workers(30, cpu_count=12) == 12
    assert MPBParallelConfig(workers="auto").resolved_workers(4, cpu_count=64) == 4
    assert MPBParallelConfig(workers=99).resolved_workers(30, cpu_count=64) == 20


def test_workers_two_runs_jobs_concurrently(tmp_path: Path) -> None:
    started = time.perf_counter()
    results, summary = run_mpb_worker_jobs(
        _jobs(4),
        config=MPBParallelConfig(workers=2, min_workers=1, max_workers=2, worker_backend="thread"),
        shard_dir=tmp_path / "worker_shards",
        solver_fn=sleepy_fake_solver,
    )
    elapsed = time.perf_counter() - started

    assert summary["workers_resolved"] == 2
    assert summary["suppress_solver_output"] is True
    assert summary["completed_count"] == 4
    assert [result.sample_index for result in results] == [0, 1, 2, 3]
    assert elapsed < 0.55
    assert (tmp_path / "worker_summary.json").is_file()
    assert (tmp_path / "worker_shards" / "unit_000000.json").is_file()


def test_failure_is_recorded_and_resume_skips_completed_shards(tmp_path: Path) -> None:
    config = MPBParallelConfig(workers=1, min_workers=1, max_workers=2, worker_backend="thread", fail_fast=False)
    results, summary = run_mpb_worker_jobs(_jobs(3), config=config, shard_dir=tmp_path / "worker_shards", solver_fn=failing_fake_solver)

    assert summary["completed_count"] == 2
    assert summary["failed_count"] == 1
    assert results[1].status == "solver_execution_failed"

    resumed, resumed_summary = run_mpb_worker_jobs(_jobs(3), config=config, shard_dir=tmp_path / "worker_shards", solver_fn=failing_fake_solver)

    assert resumed_summary["resumed_count"] == 2
    assert [result.sample_index for result in resumed] == [0, 1, 2]


def test_worker_dispatches_explicit_solver_input(tmp_path: Path) -> None:
    job = _jobs(1)[0]
    job = MPBWorkerJob(
        sample_id=job.sample_id,
        sample_index=job.sample_index,
        split=job.split,
        rho=job.rho,
        solver_input={"kind": "analytic"},
        k_points=job.k_points,
        solver_config=job.solver_config,
        rho_hash=job.rho_hash,
        geometry_hash=job.geometry_hash,
    )
    results, summary = run_mpb_worker_jobs(
        [job],
        config=MPBParallelConfig(workers=1, min_workers=1, max_workers=1, worker_backend="thread"),
        shard_dir=tmp_path / "worker_shards",
        solver_fn=solver_input_fake_solver,
    )

    assert summary["completed_count"] == 1
    assert results[0].status == "computed"
    assert np.all(results[0].bands == 1.0)


def test_resume_rejects_a_shard_with_mismatched_identity(tmp_path: Path) -> None:
    config = MPBParallelConfig(workers=1, min_workers=1, max_workers=1, worker_backend="thread")
    shard_dir = tmp_path / "worker_shards"
    run_mpb_worker_jobs(_jobs(2), config=config, shard_dir=shard_dir, solver_fn=sleepy_fake_solver)
    payload_path = shard_dir / "unit_000000.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["geometry_hash"] = "wrong-geometry"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    _, summary = run_mpb_worker_jobs(_jobs(2), config=config, shard_dir=shard_dir, solver_fn=sleepy_fake_solver)

    assert summary["resumed_count"] == 1
    repaired = json.loads(payload_path.read_text(encoding="utf-8"))
    assert repaired["geometry_hash"] == "geom0"


def test_partition_runner_uses_one_hdf5_and_resumes_successful_cases(tmp_path: Path) -> None:
    h5py = __import__("pytest").importorskip("h5py")
    output = tmp_path / "shard_000" / "partition.h5"
    identity = {
        "shard_index": 0,
        "shard_count": 1,
        "case_start": 0,
        "case_end": 4,
        "plan_sha256": "a" * 64,
    }
    config = MPBParallelConfig(
        workers=2,
        min_workers=1,
        max_workers=2,
        worker_backend="thread",
        fail_fast=False,
    )
    summary = run_mpb_worker_partition(
        _jobs(4),
        config=config,
        partition_path=output,
        identity=identity,
        solver_fn=sleepy_fake_solver,
    )

    assert summary["completed_count"] == 4
    assert summary["failed_count"] == 0
    assert not list(tmp_path.rglob("*.npy"))
    with h5py.File(output, "r") as handle:
        assert bool(handle.attrs["partition_complete"])
        assert handle["bands"].shape == (4, 2, 2)
        assert np.all(np.asarray(handle["success"]) == 1)

    resumed = run_mpb_worker_partition(
        _jobs(4),
        config=config,
        partition_path=output,
        identity=identity,
        allow_resume=True,
        solver_fn=sleepy_fake_solver,
    )
    assert resumed["resumed_count"] == 4
    assert resumed["computed_this_run"] == 0


def test_partition_resume_rejects_identity_change(tmp_path: Path) -> None:
    __import__("pytest").importorskip("h5py")
    output = tmp_path / "partition.h5"
    config = MPBParallelConfig(workers=1, min_workers=1, max_workers=1, worker_backend="thread")
    run_mpb_worker_partition(
        _jobs(1),
        config=config,
        partition_path=output,
        identity={"plan_sha256": "a" * 64},
        solver_fn=sleepy_fake_solver,
    )
    import pytest

    with pytest.raises(ValueError, match="identity"):
        run_mpb_worker_partition(
            _jobs(1),
            config=config,
            partition_path=output,
            identity={"plan_sha256": "b" * 64},
            allow_resume=True,
            solver_fn=sleepy_fake_solver,
        )
