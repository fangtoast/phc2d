"""Generate one independent shard of a V2 physical-k query dataset.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data.lift import lift_rho_to_grid
from phc2d.generators import build_structure_generator, collect_structure_metadata
from phc2d.solvers import MPBWorkerJob, MpbBandSolverConfig, run_mpb_worker_jobs, solve_bands_for_rho_mpb
from scripts.run_query_dataset_v2_pilot_generation import (
    ROOT,
    _build_k_path_spec,
    _display_path,
    _effective_split_counts,
    _git_commit,
    _hash_array,
    _jsonable,
    _resolve_n_wedge,
    _resolve_split_counts,
    _sha256_text,
    _solver_parallel_config,
    _split_labels,
    _write_json,
    _write_jsonl,
    derive_grid_shape,
    load_query_pilot_config,
)


def _metadata_jsonl(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) for row in rows)


def generate_shard(
    config_path: str | Path,
    *,
    start: int,
    end: int,
    out: str | Path,
    run_dir: str | Path | None = None,
    workers: int | str | None = 1,
    worker_backend: str | None = None,
    solver_fn=solve_bands_for_rho_mpb,
) -> Path:
    """Generate samples in ``[start, end)`` as an independently runnable shard."""

    config_path = Path(config_path)
    cfg = load_query_pilot_config(config_path)
    dataset_cfg = dict(cfg.get("dataset") or {})
    geometry_cfg = dict(cfg.get("geometry") or {})
    solver_cfg = dict(cfg.get("solver") or {})
    experiment_id = str(cfg.get("experiment_id") or Path(str(dataset_cfg.get("output_path", "query_dataset_v2"))).stem)
    sample_count = int(dataset_cfg["sample_count"])
    start = int(start)
    end = int(end)
    if start < 0 or end <= start or end > sample_count:
        raise ValueError(f"Shard range must satisfy 0 <= start < end <= {sample_count}, got [{start}, {end})")

    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    run_root = Path(run_dir) if run_dir is not None else out_path.parent / f"{out_path.stem}_run"
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    resolved_yaml = yaml.safe_dump(_jsonable(cfg), sort_keys=False, allow_unicode=True)
    config_hash = _sha256_text(resolved_yaml)
    git_commit = _git_commit()
    grid_shape = derive_grid_shape(geometry_cfg)
    n_wedge = _resolve_n_wedge(geometry_cfg, grid_shape)
    full_split_counts = _resolve_split_counts(dataset_cfg)
    labels = _split_labels(_effective_split_counts(full_split_counts, sample_count))
    k_path_spec = _build_k_path_spec(
        cfg,
        config_path=config_path,
        config_hash=config_hash,
        run_id=run_root.name,
        status="shard_generated",
        git_commit=git_commit,
    )
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
        worker_backend=worker_backend,
        force_serial_default=solver_fn is not solve_bands_for_rho_mpb,
    )

    rho_rows: list[np.ndarray] = []
    jobs: list[MPBWorkerJob] = []
    sample_plan: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sample_index in range(end):
        sample = generator.sample(rng=rng, dataset_cfg=generator_cfg)
        if sample_index < start:
            continue
        rho = np.asarray(sample.rho, dtype=np.float32).reshape(-1)
        if int(rho.shape[0]) != n_wedge:
            raise ValueError(f"Generated rho length {rho.shape[0]} does not match config-derived n_wedge={n_wedge}")
        binary_grid = np.asarray(sample.binary_grid, dtype=np.float32)
        if tuple(int(v) for v in binary_grid.shape) != grid_shape:
            binary_grid = lift_rho_to_grid(rho, grid_shape=grid_shape).astype(np.float32)
        rho_hash = _hash_array(rho)
        geometry_hash = _hash_array((binary_grid > 0.5).astype(np.uint8))
        split = labels[sample_index]
        sample_id = f"{experiment_id}_{sample_index:06d}"
        structure_meta = collect_structure_metadata(
            binary_grid=binary_grid,
            seed=int(dataset_cfg.get("seed", 0)) + sample_index,
            sample_size=int(grid_shape[0]),
            generator_name=str(getattr(generator, "name", "unknown")),
            symmetry=str(sample.metadata.get("symmetry", "")),
            postprocess_steps=list(sample.postprocess_steps),
        )
        plan_row = {
            "sample_id": sample_id,
            "split": split,
            "sample_index": sample_index,
            "rho_hash": rho_hash,
            "geometry_hash": geometry_hash,
            "solver_provenance_ref": "solver_provenance.json",
            "k_path_sidecar_ref": "k_path_sidecar.json",
            "generation_status": "shard_generated",
            "structure_metadata": {**structure_meta, **dict(sample.metadata)},
        }
        sample_plan.append(plan_row)
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

    _write_jsonl(run_root / "sample_plan.jsonl", sample_plan)
    worker_results, worker_summary = run_mpb_worker_jobs(
        jobs,
        config=parallel_cfg,
        shard_dir=run_root / "worker_shards",
        solver_fn=solver_fn,
    )
    failures = [result for result in worker_results if result.status != "computed" or result.bands is None]
    if failures:
        _write_json(
            run_root / "results.json",
            {
                "action": "query_dataset_v2_shard_generation",
                "status": "solver_execution_failed",
                "failed_sample_id": failures[0].sample_id,
                "reason": failures[0].error,
            },
        )
        raise RuntimeError(f"Shard generation failed at {failures[0].sample_id}: {failures[0].error}")

    result_by_index = {int(result.sample_index): result for result in worker_results}
    wh_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for row, rho in zip(sample_plan, rho_rows, strict=True):
        result = result_by_index[int(row["sample_index"])]
        wh = np.asarray(result.bands, dtype=np.float32)
        wh_rows.append(wh)
        metadata_rows.append({**row, "Wh_hash": result.Wh_hash, "mpb_solve_metadata": result.solve_metadata})
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

    rho_array = np.stack(rho_rows, axis=0).astype(np.float32)
    wh_array = np.stack(wh_rows, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rho=rho_array,
        Wh=wh_array,
        sample_index=np.arange(start, end, dtype=np.int64),
        sample_id=np.asarray([row["sample_id"] for row in metadata_rows], dtype=str),
        split=np.asarray([row["split"] for row in metadata_rows], dtype=str),
        metadata_jsonl=_metadata_jsonl(metadata_rows),
        per_sample_jsonl=_metadata_jsonl(per_sample_rows),
        config_yaml=resolved_yaml,
        config_hash=config_hash,
    )
    wall_time = float(time.perf_counter() - started)
    _write_json(
        run_root / "metrics.json",
        {
            "physical_validation_decision": "shard_generated",
            "real_mpb_generation_started": True,
            "real_mpb_generation_completed": True,
            "wall_time_total_s": wall_time,
            "sample_count": int(end - start),
            "sample_start": start,
            "sample_end": end,
            "mpb_generation_time_s": float(sum(float(row["mpb_solve_time_s"]) for row in per_sample_rows)),
            **worker_summary,
        },
    )
    _write_json(
        run_root / "results.json",
        {
            "action": "query_dataset_v2_shard_generation",
            "status": "shard_generated",
            "shard_path": _display_path(out_path),
            "sample_start": start,
            "sample_end": end,
            "generated_sample_count": int(end - start),
        },
    )
    return out_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Generate one independent Query Dataset V2 MPB shard.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--workers", default=1)
    parser.add_argument("--worker-backend", default=None)
    args = parser.parse_args(argv)
    workers: int | str | None = str(args.workers)
    if workers.lower() != "auto":
        workers = int(workers)
    return generate_shard(
        args.config,
        start=args.start,
        end=args.end,
        out=args.out,
        run_dir=args.run_dir,
        workers=workers,
        worker_backend=args.worker_backend,
    )


if __name__ == "__main__":
    main()
