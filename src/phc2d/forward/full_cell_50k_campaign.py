"""Run the authorized 63-wave full-cell 50k forward training campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml

from phc2d.forward.full_cell_wave import REQUIRED_RUN_FILES, aggregate_full_cell_forward_wave

MODELS = ("mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2")
SEEDS = (0, 1, 2)
TRAIN_SIZES = (40000, 20000, 8000)
EPOCH_BUDGETS = (20, 100, 200)
REPRESENTATIONS = (
    ("binary_native", "configs/research/full_cell_forward_binary_native_50000_v1.yaml", None),
    (
        "graded_continuous",
        "configs/research/full_cell_forward_graded_continuous_50000_v1.yaml",
        None,
    ),
    ("graded_t03", "configs/research/full_cell_forward_graded_threshold_50000_v1.yaml", 0.3),
    ("graded_t04", "configs/research/full_cell_forward_graded_threshold_50000_v1.yaml", 0.4),
    ("graded_t05", "configs/research/full_cell_forward_graded_threshold_50000_v1.yaml", 0.5),
    ("graded_t06", "configs/research/full_cell_forward_graded_threshold_50000_v1.yaml", 0.6),
    ("graded_t07", "configs/research/full_cell_forward_graded_threshold_50000_v1.yaml", 0.7),
)
TERMINAL_FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}

SlurmReader = Callable[[Sequence[str]], Mapping[str, Mapping[str, str]]]
Submitter = Callable[[Mapping[str, Any]], Path]
Aggregator = Callable[[Path, Path], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON mapping: {path}")
    return dict(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    expected = {(model, str(seed)) for model in MODELS for seed in SEEDS}
    actual = {(row.get("model", ""), row.get("seed", "")) for row in rows}
    if len(rows) != 12 or actual != expected:
        raise ValueError(f"manifest must contain exactly 4 models x 3 seeds: {path}")
    job_ids = [row.get("job_id", "") for row in rows]
    if any(not value.isdigit() for value in job_ids) or len(set(job_ids)) != 12:
        raise ValueError(f"manifest has invalid or duplicate Slurm job IDs: {path}")
    return rows


def build_wave_plan() -> list[dict[str, Any]]:
    """Return the frozen 63-wave, 756-run execution order."""

    waves: list[dict[str, Any]] = []
    for train_size in TRAIN_SIZES:
        for representation, config, threshold in REPRESENTATIONS:
            for max_epochs in EPOCH_BUDGETS:
                key = f"{representation}_n{train_size}_e{max_epochs}"
                waves.append(
                    {
                        "index": len(waves) + 1,
                        "key": key,
                        "wave_id": f"fc50k_{key}",
                        "representation": representation,
                        "config": config,
                        "threshold": threshold,
                        "resolution": 64,
                        "train_size": train_size,
                        "max_epochs": max_epochs,
                        "planned_runs": 12,
                        "status": "planned",
                        "manifest": None,
                        "job_ids": [],
                        "aggregation_dir": None,
                    }
                )
    if len(waves) != 63 or sum(int(wave["planned_runs"]) for wave in waves) != 756:
        raise RuntimeError("internal error: the frozen campaign is not 63 waves / 756 runs")
    return waves


def initialize_campaign(
    campaign_dir: str | Path,
    *,
    repository_root: str | Path = ".",
    adopted_manifests: Mapping[str, str | Path] | None = None,
    authorization_id: str,
) -> Path:
    """Create one immutable campaign state and adopt already submitted waves."""

    root = Path(repository_root).resolve()
    out = Path(campaign_dir)
    if not out.is_absolute():
        out = root / out
    state_path = out / "campaign_status.json"
    if state_path.exists():
        raise FileExistsError(f"campaign already initialized: {state_path}")
    waves = build_wave_plan()
    by_key = {str(wave["key"]): wave for wave in waves}
    for key, raw_manifest in dict(adopted_manifests or {}).items():
        if key not in by_key:
            raise ValueError(f"cannot adopt unknown wave: {key}")
        manifest = Path(raw_manifest)
        if not manifest.is_absolute():
            manifest = root / manifest
        rows = _read_manifest(manifest)
        wave = by_key[key]
        wave["status"] = "submitted"
        wave["manifest"] = str(manifest.relative_to(root))
        wave["job_ids"] = [row["job_id"] for row in rows]
        wave["adopted_at"] = _now()
    state: dict[str, Any] = {
        "schema_version": "phc2d.full_cell_forward_50k_campaign.v1",
        "status": "campaign_running",
        "authorization_id": authorization_id,
        "formal_submission_authorized": True,
        "created_at": _now(),
        "updated_at": _now(),
        "planned_waves": 63,
        "planned_runs": 756,
        "completed_waves": 0,
        "completed_runs": 0,
        "failed_runs": 0,
        "active_wave": None,
        "next_wave": None,
        "waves": waves,
        "next_gate": "audit_active_wave_then_submit_next",
    }
    _refresh_counts(state)
    _atomic_write_json(state_path, state)
    return state_path


def _validate_run_artifacts(root: Path, row: Mapping[str, str], wave: Mapping[str, Any]) -> list[str]:
    run_dir = Path(str(row["run_dir"]))
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    failures = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if failures:
        return [f"missing:{name}" for name in failures]
    try:
        results = _read_json(run_dir / "results.json")
        metrics = _read_json(run_dir / "metrics.json")
        meta = _read_json(run_dir / "run.meta.json")
        with (run_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
            history = [dict(item) for item in csv.DictReader(handle)]
        resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser errors vary
        return [f"parse_error:{type(exc).__name__}:{exc}"]
    try:
        checks = {
            "results_status": results.get("status") == "formal_training_success",
            "test_not_executed": not bool(results.get("test_executed")),
            "model": str(results.get("model")) == str(row["model"]),
            "seed": int(results.get("seed", -1)) == int(row["seed"]),
            "train_size": int(results.get("train_size", -1)) == int(wave["train_size"]),
            "max_epochs": int(results.get("max_epochs", -1)) == int(wave["max_epochs"]),
            "history_length": len(history) == int(wave["max_epochs"]),
            "history_terminal_epoch": bool(history)
            and int(history[-1].get("epoch", -1)) == int(wave["max_epochs"]),
            "checkpoint_hash": len(str(results.get("checkpoint_sha256", ""))) == 64,
            "metric_finite": math.isfinite(
                float(metrics.get("best_validation_bands_2_10_mre_pct", math.nan))
            ),
            "resolved_train_size": int(resolved["training"]["train_size"])
            == int(wave["train_size"]),
            "resolved_max_epochs": int(resolved["training"]["max_epochs"])
            == int(wave["max_epochs"]),
            "dataset_hash": len(str(meta.get("dataset_sha256", ""))) == 64,
            "subset_hash": len(str(meta.get("train_subset_hash", ""))) == 64,
        }
    except Exception as exc:
        return [f"contract_error:{type(exc).__name__}:{exc}"]
    return [name for name, passed in checks.items() if not passed]


def audit_wave(
    wave: Mapping[str, Any],
    *,
    repository_root: str | Path,
    slurm_reader: SlurmReader,
) -> dict[str, Any]:
    """Audit scheduler terminal state and every standard artifact for one wave."""

    root = Path(repository_root).resolve()
    manifest_value = wave.get("manifest")
    if not manifest_value:
        return {"status": "planned", "completed": 0, "failed": 0, "active": 0, "unknown": 0}
    manifest = Path(str(manifest_value))
    if not manifest.is_absolute():
        manifest = root / manifest
    try:
        rows = _read_manifest(manifest)
    except Exception as exc:
        return {
            "status": "failed_manifest",
            "completed": 0,
            "failed": 12,
            "active": 0,
            "unknown": 0,
            "failures": [str(exc)],
        }
    scheduler = dict(slurm_reader([row["job_id"] for row in rows]))
    completed = 0
    failed = 0
    active = 0
    unknown = 0
    failures: list[str] = []
    for row in rows:
        job_id = row["job_id"]
        state = str(dict(scheduler.get(job_id, {})).get("state", "UNKNOWN")).split()[0].upper()
        if state == "COMPLETED":
            completed += 1
        elif state in TERMINAL_FAILURE_STATES:
            failed += 1
            failures.append(f"job:{job_id}:{state}")
        elif state == "UNKNOWN":
            unknown += 1
        else:
            active += 1
    if failed:
        return {
            "status": "failed_slurm",
            "completed": completed,
            "failed": failed,
            "active": active,
            "unknown": unknown,
            "failures": failures,
        }
    if completed != 12:
        return {
            "status": "running",
            "completed": completed,
            "failed": 0,
            "active": active,
            "unknown": unknown,
            "failures": [],
        }
    for row in rows:
        run_failures = _validate_run_artifacts(root, row, wave)
        failures.extend(f"run:{row['model']}:seed{row['seed']}:{item}" for item in run_failures)
    failed_runs = len({tuple(item.split(":", 3)[1:3]) for item in failures})
    return {
        "status": "complete" if not failures else "failed_artifacts",
        "completed": completed if not failures else completed - failed_runs,
        "failed": failed_runs,
        "active": 0,
        "unknown": 0,
        "failures": failures,
    }


def _default_slurm_reader(job_ids: Sequence[str]) -> Mapping[str, Mapping[str, str]]:
    command = [
        "sacct",
        "-X",
        "-n",
        "-j",
        ",".join(job_ids),
        "-o",
        "JobIDRaw,State,ExitCode,Elapsed,NodeList",
        "-P",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: dict[str, dict[str, str]] = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 5 or fields[0] not in job_ids:
            continue
        rows[fields[0]] = {
            "state": fields[1],
            "exit_code": fields[2],
            "elapsed": fields[3],
            "node": fields[4],
        }
    return rows


def _default_submitter(root: Path) -> Submitter:
    def submit(wave: Mapping[str, Any]) -> Path:
        threshold = "" if wave.get("threshold") is None else str(wave["threshold"])
        env = dict(os.environ)
        env.update(
            {
                "PYTHONPATH": "src",
                "PHC2D_WAVE_CONCURRENCY": "12",
                "PHC2D_FORWARD_TIME_LIMIT": "12:00:00",
            }
        )
        command = [
            "bash",
            "jobs/paracloud_submit_full_cell_forward_wave.sh",
            "train",
            str(wave["config"]),
            str(wave["wave_id"]),
            threshold,
            str(wave["resolution"]),
            str(wave["train_size"]),
            str(wave["max_epochs"]),
        ]
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("submission produced no manifest path")
        manifest = Path(lines[-1])
        return manifest if manifest.is_absolute() else root / manifest

    return submit


def _default_aggregator(root: Path) -> Aggregator:
    def aggregate(manifest: Path, output_dir: Path) -> None:
        aggregate_full_cell_forward_wave(manifest, output_dir=output_dir, repository_root=root)

    return aggregate


def _refresh_counts(state: dict[str, Any]) -> None:
    waves = list(state["waves"])
    completed_waves = sum(str(wave["status"]) == "complete" for wave in waves)
    state["completed_waves"] = completed_waves
    state["completed_runs"] = completed_waves * 12
    state["failed_runs"] = sum(int(dict(wave.get("audit") or {}).get("failed", 0)) for wave in waves)
    active = [str(wave["key"]) for wave in waves if str(wave["status"]) in {"submitted", "submitting"}]
    planned = [str(wave["key"]) for wave in waves if str(wave["status"]) == "planned"]
    state["active_wave"] = active[0] if len(active) == 1 else None
    state["next_wave"] = planned[0] if planned else None
    state["updated_at"] = _now()


@contextmanager
def _campaign_lock(campaign_dir: Path) -> Iterator[None]:
    lock_path = campaign_dir / "advance.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, f"pid={os.getpid()} time={_now()}\n".encode("utf-8"))
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)


def advance_campaign(
    campaign_dir: str | Path,
    *,
    repository_root: str | Path = ".",
    submit: bool,
    slurm_reader: SlurmReader | None = None,
    submitter: Submitter | None = None,
    aggregator: Aggregator | None = None,
) -> dict[str, Any]:
    """Audit the active wave and submit at most one next authorized wave."""

    root = Path(repository_root).resolve()
    out = Path(campaign_dir)
    if not out.is_absolute():
        out = root / out
    state_path = out / "campaign_status.json"
    reader = slurm_reader or _default_slurm_reader
    submit_one = submitter or _default_submitter(root)
    aggregate = aggregator or _default_aggregator(root)
    with _campaign_lock(out):
        state = _read_json(state_path)
        if not bool(state.get("formal_submission_authorized")):
            raise PermissionError("campaign lacks explicit formal submission authorization")
        if str(state.get("status")) in {"campaign_complete", "blocked_failed_wave", "blocked_submission_failure"}:
            return state
        interrupted_submissions = [
            str(wave["key"]) for wave in state["waves"] if str(wave["status"]) == "submitting"
        ]
        if interrupted_submissions:
            state["status"] = "blocked_submission_failure"
            state["submission_recovery_required"] = interrupted_submissions
            state["next_gate"] = "manual_partial_submission_audit_no_automatic_retry"
            _refresh_counts(state)
            _atomic_write_json(state_path, state)
            return state

        active_after_audit: list[dict[str, Any]] = []
        for wave in state["waves"]:
            if str(wave["status"]) != "submitted":
                continue
            audit = audit_wave(wave, repository_root=root, slurm_reader=reader)
            wave["audit"] = audit
            wave["last_checked_at"] = _now()
            if audit["status"] == "complete":
                manifest = root / str(wave["manifest"])
                aggregation_dir = out / "aggregations" / str(wave["key"])
                try:
                    if not (aggregation_dir / "wave.meta.json").is_file():
                        aggregate(manifest, aggregation_dir)
                except Exception as exc:
                    audit["status"] = "failed_aggregation"
                    audit["failures"] = [f"{type(exc).__name__}:{exc}"]
                else:
                    wave["status"] = "complete"
                    wave["aggregation_dir"] = str(aggregation_dir.relative_to(root))
                    wave["completed_at"] = _now()
            if str(audit["status"]).startswith("failed"):
                wave["status"] = "failed"
                state["status"] = "blocked_failed_wave"
                state["next_gate"] = "manual_failure_audit_no_automatic_retry"
            elif wave["status"] == "submitted":
                active_after_audit.append(wave)

        if state["status"] == "blocked_failed_wave":
            _refresh_counts(state)
            _atomic_write_json(state_path, state)
            return state
        if len(active_after_audit) > 1:
            state["status"] = "blocked_failed_wave"
            state["next_gate"] = "manual_audit_multiple_active_waves"
            _refresh_counts(state)
            _atomic_write_json(state_path, state)
            return state

        if not active_after_audit:
            planned = [wave for wave in state["waves"] if wave["status"] == "planned"]
            if not planned:
                state["status"] = "campaign_complete"
                state["completed_at"] = _now()
                state["next_gate"] = "forward_training_complete_pending_evaluation"
            elif submit:
                wave = planned[0]
                wave["status"] = "submitting"
                wave["submission_started_at"] = _now()
                _refresh_counts(state)
                _atomic_write_json(state_path, state)
                try:
                    manifest = submit_one(wave).resolve()
                    rows = _read_manifest(manifest)
                except Exception as exc:
                    wave["status"] = "failed"
                    wave["submission_error"] = f"{type(exc).__name__}:{exc}"
                    state["status"] = "blocked_submission_failure"
                    state["next_gate"] = "manual_partial_submission_audit_no_automatic_retry"
                else:
                    wave["status"] = "submitted"
                    wave["manifest"] = str(manifest.relative_to(root))
                    wave["job_ids"] = [row["job_id"] for row in rows]
                    wave["submitted_at"] = _now()
                    state["next_gate"] = "audit_active_wave_then_submit_next"

        _refresh_counts(state)
        _atomic_write_json(state_path, state)
        return state


def _parse_adoptions(values: Sequence[str]) -> dict[str, Path]:
    adopted: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--adopt must use WAVE_KEY=MANIFEST")
        key, raw_path = value.split("=", 1)
        adopted[key] = Path(raw_path)
    return adopted


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--campaign-dir", type=Path, required=True)
    init_parser.add_argument("--repository-root", type=Path, default=Path("."))
    init_parser.add_argument("--authorization-id", required=True)
    init_parser.add_argument("--adopt", action="append", default=[])
    advance_parser = subparsers.add_parser("advance")
    advance_parser.add_argument("--campaign-dir", type=Path, required=True)
    advance_parser.add_argument("--repository-root", type=Path, default=Path("."))
    advance_parser.add_argument("--submit", action="store_true")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--campaign-dir", type=Path, required=True)
    status_parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.command == "init":
        print(
            initialize_campaign(
                args.campaign_dir,
                repository_root=args.repository_root,
                adopted_manifests=_parse_adoptions(args.adopt),
                authorization_id=args.authorization_id,
            )
        )
    elif args.command == "advance":
        state = advance_campaign(
            args.campaign_dir,
            repository_root=args.repository_root,
            submit=bool(args.submit),
        )
        print(json.dumps(state, ensure_ascii=False, sort_keys=True))
    else:
        root = Path(args.repository_root).resolve()
        path = Path(args.campaign_dir)
        if not path.is_absolute():
            path = root / path
        print(json.dumps(_read_json(path / "campaign_status.json"), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["advance_campaign", "audit_wave", "build_wave_plan", "initialize_campaign"]
