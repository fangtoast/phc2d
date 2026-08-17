"""Multi-stage HPC adapter for full-cell MPB dataset campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.data.full_cell import write_full_cell_hdf5_streaming
from phc2d.data.full_cell_generation import (
    FullCellSolverFn,
    _effective_split_counts,
    _file_hash,
    _load_frozen_sample_plan,
    _relative,
    _solver_config,
    _stable_stratified_splits,
    _validate_execution_boundary,
    load_full_cell_generation_config,
    prepare_full_cell_sample_plan,
)
from phc2d.hpc.campaign import collect_campaign_status
from phc2d.hpc.contracts import (
    PhysicalCampaignManifest,
    PhysicalPlanManifest,
    PhysicalShardSpec,
    atomic_write_json,
    sha256_file,
    sha256_payload,
    validate_shard_coverage,
)
from phc2d.runs import write_metrics, write_results, write_run_meta
from phc2d.solvers import (
    MPBParallelConfig,
    MPBWorkerJob,
    run_mpb_worker_partition,
    solve_bands_for_full_cell_mpb,
)


class FullCellHPCAdapter:
    """Runtime adapter exposing full-cell stages through the shared HPC protocol."""

    adapter_id = "full_cell_band"

    def plan(
        self,
        config_path: str | Path,
        *,
        run_dir: str | Path,
        workers: int | str | None,
        shard_count: int,
        platform_id: str,
    ) -> Path:
        return prepare_full_cell_hpc_plan(
            config_path,
            run_dir=run_dir,
            workers=workers,
            shard_count=shard_count,
            platform_id=platform_id,
        )

    def solve_shard(
        self,
        config_path: str | Path,
        *,
        run_dir: str | Path,
        shard_index: int,
        shard_count: int,
        workers: int | str | None,
        allow_resume: bool,
        skip_environment_check: bool,
        **kwargs: Any,
    ) -> Path:
        return solve_full_cell_hpc_shard(
            config_path,
            run_dir=run_dir,
            shard_index=shard_index,
            shard_count=shard_count,
            workers=workers,
            allow_resume=allow_resume,
            skip_environment_check=skip_environment_check,
            **kwargs,
        )

    def finalize(
        self,
        config_path: str | Path,
        *,
        run_dir: str | Path,
        workers: int | str,
    ) -> Path:
        return finalize_full_cell_hpc(
            config_path,
            run_dir=run_dir,
            raster_workers=workers,
        )


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON mapping: {path}")
    return dict(payload)


def _full_cell_jobs(
    config_path: str | Path,
    run_dir: str | Path,
) -> tuple[dict[str, Any], list[Any], list[str], list[str], np.ndarray, np.ndarray, list[MPBWorkerJob]]:
    cfg = load_full_cell_generation_config(config_path)
    root = Path(run_dir)
    count = int(dict(cfg["dataset"])["sample_count"])
    structures, strata, split_labels = _load_frozen_sample_plan(
        root / "sample_plan.jsonl", cfg, effective_count=count
    )
    split_counts = _effective_split_counts(dict(cfg["dataset"])["split"], count, count)
    sample_ids = [item.sample_id for item in structures]
    idx_train, idx_val, idx_test, expected_labels = _stable_stratified_splits(
        sample_ids, strata, split_counts
    )
    if split_labels != expected_labels:
        raise ValueError("Frozen sample plan split assignments changed")
    from phc2d.data.full_cell import build_full_bz_grid

    k_points, k_grid_index = build_full_bz_grid(int(dict(cfg["k_grid"])["n_side"]))
    solver_cfg = _solver_config(dict(cfg["solver"]))
    jobs = [
        MPBWorkerJob(
            sample_id=structure.sample_id,
            sample_index=index,
            split=split_labels[index],
            rho=np.asarray(structure.parameter_vector, dtype=np.float32),
            solver_input=structure,
            k_points=k_points,
            solver_config=solver_cfg,
            rho_hash=sha256_payload(list(structure.parameter_vector)),
            geometry_hash=structure.geometry_hash,
        )
        for index, structure in enumerate(structures)
    ]
    return (
        cfg,
        structures,
        strata,
        split_labels,
        k_points,
        k_grid_index,
        jobs,
    )


def prepare_full_cell_hpc_plan(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    workers: int | str | None = None,
    shard_count: int,
    platform_id: str,
) -> Path:
    """Freeze one full-cell plan and publish the generic HPC manifests."""

    out = prepare_full_cell_sample_plan(config_path, run_dir=run_dir, workers=workers)
    cfg = load_full_cell_generation_config(config_path)
    plan_manifest = _read_json(out / "sample_plan_manifest.json")
    plan_sha256 = str(plan_manifest["sample_plan_sha256"])
    physical = PhysicalPlanManifest(
        adapter_id="full_cell_band",
        case_count=int(dict(cfg["dataset"])["sample_count"]),
        plan_sha256=plan_sha256,
        solver_fingerprint=sha256_payload(
            {"solver": dict(cfg["solver"]), "k_grid": dict(cfg["k_grid"])}
        ),
        output_schema="phc2d.full_cell.v1",
        case_plan="sample_plan.jsonl",
    ).to_mapping()
    physical["shard_count"] = int(shard_count)
    atomic_write_json(out / "physical_plan_manifest.json", physical)
    campaign = PhysicalCampaignManifest(
        campaign_id=out.name,
        adapter_id="full_cell_band",
        state="plan_ready",
        plan_sha256=plan_sha256,
        planned_cases=physical["case_count"],
        shard_count=int(shard_count),
        platform_id=platform_id,
        next_gate="benchmark_or_solver_array",
    )
    atomic_write_json(out / "physical_campaign_manifest.json", campaign.to_mapping())
    collect_campaign_status(out, write=True)
    return out


def solve_full_cell_hpc_shard(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    shard_index: int,
    shard_count: int,
    workers: int | str | None = None,
    allow_resume: bool = False,
    skip_environment_check: bool = False,
    solver_fn: FullCellSolverFn = solve_bands_for_full_cell_mpb,
) -> Path:
    """Solve one exclusive full-cell shard into a durable partition HDF5."""

    root = Path(run_dir)
    physical = _read_json(root / "physical_plan_manifest.json")
    if str(physical.get("adapter_id")) != "full_cell_band":
        raise ValueError("Run directory does not contain a full-cell HPC plan")
    if int(physical.get("shard_count", 0)) != int(shard_count):
        raise ValueError("shard_count differs from the frozen physical plan")
    if sha256_file(root / str(physical["case_plan"])) != str(physical["plan_sha256"]):
        raise ValueError("Frozen case plan SHA-256 mismatch")
    cfg, _structures, _strata, _labels, _points, _grid, jobs = _full_cell_jobs(
        config_path, root
    )
    if not skip_environment_check:
        problems, _profile = _validate_execution_boundary(dict(cfg["solver"]))
        if problems:
            raise RuntimeError("; ".join(problems))
    worker_payload = dict(cfg.get("workers") or {})
    if workers is not None:
        worker_payload["workers"] = workers
    parallel = MPBParallelConfig.from_mapping(worker_payload)
    resolved_workers = parallel.resolved_workers(len(jobs))
    spec = PhysicalShardSpec.partition(
        case_count=len(jobs),
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        workers=resolved_workers,
        plan_sha256=str(physical["plan_sha256"]),
    )
    shard_root = root / "shards" / f"shard_{spec.shard_index:03d}"
    identity = {
        **spec.to_mapping(),
        "adapter_id": "full_cell_band",
        "solver_fingerprint": physical["solver_fingerprint"],
        "deployed_git_commit": os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT", "unknown"),
        "real_mpb_generation_eligible": bool(
            not skip_environment_check and solver_fn is solve_bands_for_full_cell_mpb
        ),
    }
    summary = run_mpb_worker_partition(
        jobs[spec.case_start : spec.case_end],
        config=parallel,
        partition_path=shard_root / "partition.h5",
        identity=identity,
        progress_path=shard_root / "progress.json",
        allow_resume=allow_resume,
        solver_fn=solver_fn,
    )
    atomic_write_json(shard_root / "shard_manifest.json", {**identity, "summary": summary})
    collect_campaign_status(root, write=True)
    if not summary["partition_complete"]:
        raise RuntimeError(
            f"Full-cell shard {spec.shard_index} failed with {summary['failed_count']} cases"
        )
    return shard_root


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def finalize_full_cell_hpc(
    config_path: str | Path,
    *,
    run_dir: str | Path,
    raster_workers: int | str = "auto",
) -> Path:
    """Validate all partitions and atomically publish the canonical dataset."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required for full-cell finalization") from exc

    started = time.perf_counter()
    root = Path(run_dir)
    cfg, structures, strata, split_labels, k_points, k_grid_index, jobs = _full_cell_jobs(
        config_path, root
    )
    physical = _read_json(root / "physical_plan_manifest.json")
    shard_count = int(physical["shard_count"])
    workers = (
        len(os.sched_getaffinity(0))
        if str(raster_workers).lower() == "auto" and hasattr(os, "sched_getaffinity")
        else int(os.cpu_count() or 1)
        if str(raster_workers).lower() == "auto"
        else int(raster_workers)
    )
    atomic_write_json(
        root / "finalize_progress.json",
        {"status": "finalizing", "started_unix_s": time.time(), "raster_workers": workers},
    )
    specs: list[PhysicalShardSpec] = []
    partition_paths: list[Path] = []
    real_eligible = True
    for index in range(shard_count):
        shard_root = root / "shards" / f"shard_{index:03d}"
        manifest = _read_json(shard_root / "shard_manifest.json")
        spec = PhysicalShardSpec(
            shard_index=int(manifest["shard_index"]),
            shard_count=int(manifest["shard_count"]),
            case_start=int(manifest["case_start"]),
            case_end=int(manifest["case_end"]),
            workers=int(manifest["workers"]),
            plan_sha256=str(manifest["plan_sha256"]),
        )
        if spec.plan_sha256 != str(physical["plan_sha256"]):
            raise ValueError("Shard plan SHA-256 does not match the physical plan")
        summary = dict(manifest.get("summary") or {})
        if not bool(summary.get("partition_complete")) or int(summary.get("failed_count", -1)) != 0:
            raise RuntimeError(f"Shard {index} is not complete and failure-free")
        specs.append(spec)
        partition_paths.append(shard_root / "partition.h5")
        real_eligible = real_eligible and bool(manifest.get("real_mpb_generation_eligible"))
    validate_shard_coverage(specs, case_count=len(jobs))

    merge_started = time.perf_counter()
    with ExitStack() as stack:
        handles = [stack.enter_context(h5py.File(path, "r")) for path in partition_paths]
        handle_for_index: dict[int, tuple[Any, int]] = {}
        for handle in handles:
            if not bool(handle.attrs.get("partition_complete", False)):
                raise RuntimeError("Partition HDF5 is not marked complete")
            indices = np.asarray(handle["sample_index"], dtype=np.int64)
            success = np.asarray(handle["success"], dtype=np.uint8)
            if not np.all(success == 1):
                raise RuntimeError("Partition HDF5 contains incomplete cases")
            for local_row, global_index in enumerate(indices.tolist()):
                if int(global_index) in handle_for_index:
                    raise ValueError("Duplicate global case index across partitions")
                handle_for_index[int(global_index)] = (handle, local_row)
        if sorted(handle_for_index) != list(range(len(jobs))):
            raise ValueError("Partition HDF5 files do not cover the complete plan")
        merge_validation_time = float(time.perf_counter() - merge_started)

        split_counts = _effective_split_counts(
            dict(cfg["dataset"])["split"], len(jobs), len(jobs)
        )
        sample_ids = [item.sample_id for item in structures]
        idx_train, idx_val, idx_test, expected_labels = _stable_stratified_splits(
            sample_ids, strata, split_counts
        )
        if split_labels != expected_labels:
            raise ValueError("Frozen split labels changed during finalization")
        metadata = {
            "protocol": "full_cell_v1",
            "lane": dict(cfg["dataset"])["lane"],
            "sample_count": len(jobs),
            "split": split_counts,
            "strata": dict(Counter(strata)),
            "k_grid": dict(cfg["k_grid"]),
            "solver": dict(cfg["solver"]),
            "config_path": _relative(Path(config_path)),
            "config_hash": sha256_payload(cfg),
            "validation_boundary": (
                "native_cluster_real_mpb_pending_wsl_acceptance"
                if real_eligible
                else "unaccepted_test_only_injected_or_unchecked_solver"
            ),
            "physical_plan_sha256": physical["plan_sha256"],
            "shard_count": shard_count,
        }
        output_name = Path(str(dict(cfg["dataset"])["output_path"])).name
        dataset_path = root / "data" / output_name
        raster_started = time.perf_counter()

        def finalize_progress(stage: str, completed: int, total: int) -> None:
            elapsed = max(time.perf_counter() - raster_started, 1.0e-12)
            atomic_write_json(
                root / "finalize_progress.json",
                {
                    "status": "finalizing",
                    "stage": stage,
                    "completed": int(completed),
                    "total": int(total),
                    "throughput_cases_per_s": float(completed / elapsed),
                    "eta_s": float((total - completed) / max(completed / elapsed, 1.0e-12)),
                    "raster_workers": workers,
                    "updated_unix_s": time.time(),
                },
            )

        write_full_cell_hdf5_streaming(
            dataset_path,
            structures=structures,
            band_shape=(len(k_points), int(jobs[0].solver_config.num_bands)),
            band_loader=lambda index: np.asarray(
                handle_for_index[index][0]["bands"][handle_for_index[index][1]],
                dtype=np.float32,
            ),
            k_points=k_points,
            k_grid_index=k_grid_index,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            raster_resolutions=tuple(int(item) for item in dict(cfg["dataset"])["raster_resolutions"]),
            metadata=metadata,
            raster_workers=workers,
            progress_fn=finalize_progress,
            raster_method=str(
                dict(dict(cfg["dataset"]).get("raster_contract") or {}).get(
                    "method",
                    "direct_cell_center_v1",
                )
            ),
            raster_source_resolution=int(
                dict(dict(cfg["dataset"]).get("raster_contract") or {}).get(
                    "source_resolution",
                    max(int(item) for item in dict(cfg["dataset"])["raster_resolutions"]),
                )
            ),
            content_hash_quantization_levels=int(
                dict(dict(cfg["dataset"]).get("raster_contract") or {}).get(
                    "content_hash_quantization_levels",
                    65536,
                )
            ),
        )
        finalize_write_time = float(time.perf_counter() - raster_started)

        metadata_started = time.perf_counter()
        with (root / "sample_metadata.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for index, (structure, stratum) in enumerate(zip(structures, strata)):
                handle, row = handle_for_index[index]
                stream.write(
                    json.dumps(
                        {
                            "sample_id": structure.sample_id,
                            "sample_index": index,
                            "split": split_labels[index],
                            "stratum": stratum,
                            "generator_family": structure.generator_family,
                            "geometry_hash": structure.geometry_hash,
                            "Wh_hash": _decode(handle["Wh_hash"][row]),
                            "mpb_solve_time_s": float(handle["solve_time_s"][row]),
                            "solver_metadata": json.loads(
                                _decode(handle["solver_metadata_json"][row]) or "{}"
                            ),
                            "structure_stats": structure.metadata.get("stats", {}),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    (root / "failed_case_ledger.jsonl").write_text("", encoding="utf-8")
    dataset_hash = _file_hash(dataset_path)
    partition_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in partition_paths
    }
    metadata_time = float(time.perf_counter() - metadata_started)
    status = (
        "cluster_generated_pending_wsl_acceptance"
        if real_eligible
        else "test_solver_smoke_only"
    )
    provenance = {
        "backend": "meep.mpb.ModeSolver" if real_eligible else "injected_or_unchecked_solver",
        "execution_boundary": metadata["validation_boundary"],
        "real_mpb_generation_eligible": real_eligible,
        "acceptance_eligible": real_eligible,
        "physical_plan_sha256": physical["plan_sha256"],
        "shard_count": shard_count,
        "partition_sha256": partition_hashes,
        "dataset_file_hash": dataset_hash,
        "solver": dict(cfg["solver"]),
        "k_grid": dict(cfg["k_grid"]),
    }
    atomic_write_json(root / "solver_provenance.json", provenance)
    manifest = {
        "schema_version": "phc2d.full_cell_manifest.v1",
        "status": status,
        "dataset_path": str(dataset_path.relative_to(root)).replace("\\", "/"),
        "dataset_sha256": dataset_hash,
        "lane": dict(cfg["dataset"])["lane"],
        "sample_count": len(jobs),
        "Wh_shape": [len(jobs), len(k_points), int(jobs[0].solver_config.num_bands)],
        "k_grid_shape": [int(dict(cfg["k_grid"])["n_side"])] * 2,
        "raster_resolutions": list(dict(cfg["dataset"])["raster_resolutions"]),
        "split": split_counts,
        "solver_provenance": "solver_provenance.json",
        "sample_metadata": "sample_metadata.jsonl",
        "failed_case_ledger": "failed_case_ledger.jsonl",
        "physical_plan_manifest": "physical_plan_manifest.json",
        "next_gate": "download_hash_content_audit_then_wsl_cross_environment_acceptance",
    }
    _write_yaml(root / "dataset_manifest.yaml", manifest)
    elapsed = float(time.perf_counter() - started)
    write_metrics(
        root,
        {
            "status": status,
            "wall_time_total_s": elapsed,
            "shard_merge_validation_time_s": merge_validation_time,
            "finalize_hdf5_and_raster_time_s": finalize_write_time,
            "metadata_provenance_time_s": metadata_time,
            "raster_workers_resolved": workers,
            "shard_count": shard_count,
            "sample_count": len(jobs),
            "failed_count": 0,
        },
    )
    write_results(
        root,
        {
            "status": status,
            "dataset_written": True,
            "dataset_path": manifest["dataset_path"],
            "dataset_manifest": "dataset_manifest.yaml",
            "next_gate": manifest["next_gate"],
        },
    )
    write_run_meta(
        root,
        {
            "status": status,
            "validation_boundary": metadata["validation_boundary"],
            "config_path": _relative(Path(config_path)),
            "physical_hpc_adapter": "full_cell_band",
        },
    )
    with (root / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "status", "reason"])
        writer.writeheader()
        writer.writerows(
            [
                {"stage": "sample_plan", "status": "passed", "reason": "frozen plan"},
                {"stage": "solver_shards", "status": "passed", "reason": f"{shard_count} partitions"},
                {"stage": "finalize", "status": status, "reason": "atomic streaming HDF5"},
            ]
        )
    campaign = _read_json(root / "physical_campaign_manifest.json")
    campaign["state"] = "cluster_generated_pending_download"
    campaign["next_gate"] = "download_hash_content_audit"
    atomic_write_json(root / "physical_campaign_manifest.json", campaign)
    atomic_write_json(
        root / "finalize_progress.json",
        {
            "status": "completed",
            "completed": len(jobs),
            "failed": 0,
            "elapsed_s": elapsed,
            "raster_workers": workers,
            "updated_unix_s": time.time(),
        },
    )
    collect_campaign_status(root, write=True)
    return root


__all__ = [
    "FullCellHPCAdapter",
    "finalize_full_cell_hpc",
    "prepare_full_cell_hpc_plan",
    "solve_full_cell_hpc_shard",
]
