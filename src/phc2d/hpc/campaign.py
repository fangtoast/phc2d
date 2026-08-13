"""O(1)-in-sample-count monitoring and audit helpers for HPC campaigns.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from phc2d.hpc.contracts import CAMPAIGN_STATES, atomic_write_json, sha256_file


STANDARD_FINAL_ARTIFACTS = (
    "dataset_manifest.yaml",
    "metrics.json",
    "results.json",
    "history.csv",
    "solver_provenance.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON mapping: {path}")
    return dict(payload)


def _expected_shard_count(run_dir: Path) -> int:
    campaign_path = run_dir / "physical_campaign_manifest.json"
    if campaign_path.is_file():
        return int(_read_json(campaign_path)["shard_count"])
    plan_path = run_dir / "physical_plan_manifest.json"
    if plan_path.is_file():
        return int(_read_json(plan_path).get("shard_count", 0))
    return 0


def collect_campaign_status(run_dir: str | Path, *, write: bool = False) -> dict[str, Any]:
    """Read fixed control files without traversing sample-level outputs."""

    root = Path(run_dir)
    plan_path = root / "physical_plan_manifest.json"
    plan = _read_json(plan_path) if plan_path.is_file() else {}
    shard_count = _expected_shard_count(root)
    shard_rows: list[dict[str, Any]] = []
    for index in range(shard_count):
        progress_path = root / "shards" / f"shard_{index:03d}" / "progress.json"
        if progress_path.is_file():
            row = _read_json(progress_path)
        else:
            row = {"shard_index": index, "status": "missing", "completed": 0, "failed": 0}
        shard_rows.append(row)
    planned = int(plan.get("case_count", 0))
    completed = sum(int(row.get("completed", 0)) for row in shard_rows)
    failed = sum(int(row.get("failed", 0)) for row in shard_rows)
    latest_update = max((float(row.get("updated_unix_s", 0.0)) for row in shard_rows), default=0.0)
    throughput = sum(float(row.get("recent_throughput_cases_per_s", 0.0)) for row in shard_rows)
    partition_bytes = sum(int(row.get("partition_bytes", 0)) for row in shard_rows)
    remaining = max(0, planned - completed)
    all_shards_complete = bool(
        shard_count > 0
        and len(shard_rows) == shard_count
        and all(row.get("status") == "completed" for row in shard_rows)
        and completed == planned
        and failed == 0
    )
    final_files = {name: (root / name).is_file() for name in STANDARD_FINAL_ARTIFACTS}
    data_files = sorted((root / "data").glob("*.h5")) if (root / "data").is_dir() else []
    results = _read_json(root / "results.json") if (root / "results.json").is_file() else {}
    slurm_rows: list[dict[str, Any]] = []
    submission_path = root / "submission_manifest.json"
    if submission_path.is_file() and shutil.which("sacct"):
        submission = _read_json(submission_path)
        job_ids = [
            str(submission[key])
            for key in ("plan_job_id", "solve_array_job_id", "finalize_job_id")
            if submission.get(key)
        ]
        if job_ids:
            completed_process = subprocess.run(
                [
                    "sacct",
                    "-j",
                    ",".join(job_ids),
                    "--format=JobIDRaw,State,Elapsed,NodeList,AllocCPUS",
                    "-n",
                    "-P",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15.0,
            )
            if completed_process.returncode == 0:
                for line in completed_process.stdout.splitlines():
                    fields = line.strip().split("|")
                    if len(fields) == 5 and "." not in fields[0]:
                        slurm_rows.append(
                            {
                                "job_id": fields[0],
                                "state": fields[1],
                                "elapsed": fields[2],
                                "nodes": fields[3],
                                "allocated_cpus": int(fields[4] or 0),
                            }
                        )
    campaign = (
        _read_json(root / "physical_campaign_manifest.json")
        if (root / "physical_campaign_manifest.json").is_file()
        else {}
    )
    campaign_state = str(campaign.get("state", ""))
    if campaign_state in CAMPAIGN_STATES and campaign_state not in {"plan_ready", "solver_running"}:
        state = campaign_state
    elif results.get("status") == "cross_environment_accepted":
        state = "cross_environment_accepted"
    elif data_files and all(final_files.values()):
        state = str(results.get("status", "cluster_generated_pending_download"))
    elif (root / "finalize_progress.json").is_file():
        state = "finalizing"
    elif all_shards_complete:
        state = "solver_shards_complete"
    elif shard_rows:
        state = "solver_running"
    else:
        state = "plan_ready" if plan else "missing_plan"
    payload = {
        "schema_version": "phc2d.campaign_status.v1",
        "state": state,
        "planned": planned,
        "completed": completed,
        "failed": failed,
        "shard_count": shard_count,
        "recent_throughput_cases_per_s": throughput,
        "partition_bytes": partition_bytes,
        "eta_s": float(remaining / throughput) if throughput > 0.0 else None,
        "last_shard_update_unix_s": latest_update or None,
        "updated_unix_s": time.time(),
        "shards": shard_rows,
        "slurm": slurm_rows,
        "final_artifacts": final_files,
        "hdf5": [
            {"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size}
            for path in data_files
        ],
        "next_gate": campaign.get(
            "next_gate",
            results.get("next_gate", "finalize" if all_shards_complete else "solver_shards_complete"),
        ),
    }
    if write:
        atomic_write_json(root / "campaign_status.json", payload)
    return payload


def audit_campaign(run_dir: str | Path, *, verify_dataset_hash: bool = True) -> dict[str, Any]:
    """Audit control-plane completeness without promoting scientific acceptance."""

    root = Path(run_dir)
    status = collect_campaign_status(root)
    problems: list[str] = []
    if status["state"] not in CAMPAIGN_STATES and status["state"] != "missing_plan":
        problems.append(f"unknown campaign state: {status['state']}")
    if not (root / "physical_plan_manifest.json").is_file():
        problems.append("missing physical_plan_manifest.json")
    if int(status["failed"]) != 0:
        problems.append(f"failed case count is {status['failed']}")
    if int(status["planned"]) <= 0:
        problems.append("planned case count is missing or non-positive")
    elif int(status["completed"]) != int(status["planned"]):
        problems.append("solver shards are incomplete")
    missing = [name for name, present in status["final_artifacts"].items() if not present]
    if not status["hdf5"]:
        problems.append("missing final HDF5")
    if missing:
        problems.append("missing final artifacts: " + ", ".join(missing))
    manifest_hash = ""
    manifest_path = root / "dataset_manifest.yaml"
    if status["hdf5"] and manifest_path.is_file() and verify_dataset_hash:
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest_hash = str(dict(manifest or {}).get("dataset_sha256", ""))
        actual_hash = sha256_file(root / status["hdf5"][0]["path"])
        if manifest_hash != actual_hash:
            problems.append("dataset SHA-256 does not match dataset_manifest.yaml")
    return {
        "status": "passed" if not problems else "failed",
        "problems": problems,
        "campaign_state": status["state"],
        "dataset_sha256": manifest_hash or None,
        "scientific_acceptance_promoted": False,
    }


__all__ = ["STANDARD_FINAL_ARTIFACTS", "audit_campaign", "collect_campaign_status"]
