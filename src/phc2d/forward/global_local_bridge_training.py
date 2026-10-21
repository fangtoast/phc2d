"""Frozen-task execution and artifact gates for EXP-FWD-020.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


VARIANTS = ("baseline", "candidate")
TRAINING_MANIFEST_FIELDS = ("task_index", "case_id", "domain", "lane", "variant", "seed", "config", "run_dir")
CONTINUATION_MANIFEST_FIELDS = TRAINING_MANIFEST_FIELDS + ("source_run", "source_best_checkpoint_sha256", "source_epoch200_checkpoint_sha256")
STANDARD_TRAINING_ARTIFACTS = ("run.log", "run.meta.json", "config.resolved.yaml", "history.csv", "metrics.json", "results.json", "model.pt", "model_summary.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("EXP-FWD-020 runner config must be a mapping")
    return dict(value)


def load_training_task(manifest: str | Path, task_index: int, *, repository_root: str | Path = ".") -> dict[str, str]:
    """Select exactly one named-column frozen task and validate its lineage."""
    root = Path(repository_root).resolve()
    manifest_path = _resolve(root, manifest)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError("training manifest is empty")
    missing = [field for field in TRAINING_MANIFEST_FIELDS if field not in rows[0]]
    if missing:
        raise RuntimeError(f"training manifest fields missing: {missing}")
    selected = [row for row in rows if row.get("task_index") == str(task_index)]
    if len(selected) != 1:
        raise RuntimeError(f"task index must resolve to exactly one row: {task_index}")
    row = selected[0]
    if int(row["seed"]) not in {0, 1, 2}:
        raise RuntimeError(f"unsupported task seed: {row['seed']}")
    if row["lane"] != ("binary" if row["domain"].endswith("binary") else "graded"):
        raise RuntimeError(f"task lane/domain identity mismatch: {row['case_id']}")
    if row["case_id"] != f"{row['domain']}_{row['variant']}_s{row['seed']}" or row["variant"] not in VARIANTS:
        raise RuntimeError(f"task case identity mismatch: {row['case_id']}")
    config_path, run_dir = _resolve(root, row["config"]), _resolve(root, row["run_dir"])
    if not config_path.is_file() or not str(run_dir):
        raise RuntimeError(f"task config/run directory is invalid: {row['case_id']}")
    config = _load(config_path)
    experiment = dict(config.get("experiment") or {})
    if str(experiment.get("experiment_id")) != "EXP-FWD-020":
        raise RuntimeError(f"task experiment identity mismatch: {row['case_id']}")
    for key in ("domain", "lane", "variant"):
        if str(experiment.get(key)) != row[key]:
            raise RuntimeError(f"task config {key} identity mismatch: {row['case_id']}")
    if "global_local_band_operator_v1" not in dict(config.get("models") or {}):
        raise RuntimeError(f"task model identity mismatch: {row['case_id']}")
    continuation = dict(config.get("continuation") or {})
    if continuation:
        missing = [field for field in CONTINUATION_MANIFEST_FIELDS if not row.get(field)]
        if missing:
            raise RuntimeError(f"continuation manifest fields missing: {missing}")
        source = _resolve(root, row["source_run"])
        if _resolve(root, str(continuation.get("source_run", ""))) != source:
            raise RuntimeError(f"continuation source run mismatch: {row['case_id']}")
        best, epoch200 = source / "model.pt", source / "epoch200.pt"
        if not best.is_file() or _sha256(best) != row["source_best_checkpoint_sha256"] or not epoch200.is_file() or _sha256(epoch200) != row["source_epoch200_checkpoint_sha256"]:
            raise RuntimeError(f"continuation source checkpoint hash mismatch: {row['case_id']}")
        if str(continuation.get("source_experiment")) != "EXP-FWD-020" or int(continuation.get("source_epoch", -1)) != 200:
            raise RuntimeError(f"continuation source identity mismatch: {row['case_id']}")
        result_path = source / "results.json"
        if not result_path.is_file():
            raise RuntimeError(f"continuation source result missing: {row['case_id']}")
        source_result = json.loads(result_path.read_text(encoding="utf-8"))
        if source_result.get("model") != "global_local_band_operator_v1" or int(source_result.get("seed", -1)) != int(row["seed"]):
            raise RuntimeError(f"continuation source model/seed identity mismatch: {row['case_id']}")
    return {**row, "_config_path": str(config_path), "_run_dir": str(run_dir), "_is_continuation": str(bool(continuation))}


def _finite_json(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    return not isinstance(value, float) or bool(np.isfinite(value))


def audit_training_task_artifacts(row: Mapping[str, str], *, admission_probe: bool) -> None:
    """Require terminal artifacts and preserve a non-converged epoch-300 outcome."""
    run = Path(row["_run_dir"])
    required = list(STANDARD_TRAINING_ARTIFACTS)
    if row["_is_continuation"] == "True" and not admission_probe:
        required.append("epoch300.pt")
    missing = [name for name in required if not (run / name).is_file() or not (run / name).stat().st_size]
    if missing:
        raise RuntimeError(f"incomplete training artifacts for {row['case_id']}: {missing}")
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if not _finite_json(result) or not _finite_json(metrics):
        raise RuntimeError(f"non-finite training artifact: {row['case_id']}")
    if result.get("model") != "global_local_band_operator_v1" or int(result.get("seed", -1)) != int(row["seed"]):
        raise RuntimeError(f"completed task model/seed identity mismatch: {row['case_id']}")
    if result.get("test_executed") or result.get("selection_split") != "validation":
        raise RuntimeError(f"held-out or non-validation task artifact: {row['case_id']}")
    if admission_probe:
        if result.get("status") != "full_cell_forward_admission_probe_pass":
            raise RuntimeError(f"admission probe failed: {row['case_id']}")
        if row["_is_continuation"] == "True" and int(result.get("stop_epoch", -1)) != 201:
            raise RuntimeError(f"continuation admission did not execute epoch 201: {row['case_id']}")
    elif row["_is_continuation"] == "True":
        if result.get("status") not in {"continuation_training_success", "training_complete_convergence_not_met"}:
            raise RuntimeError(f"invalid continuation terminal status: {row['case_id']}")
        if int(result.get("stop_epoch", -1)) != 300 or int(result.get("history_epoch_count", -1)) != 300:
            raise RuntimeError(f"continuation did not complete epoch 300: {row['case_id']}")
        if result.get("terminal_checkpoint_sha256") != _sha256(run / "epoch300.pt"):
            raise RuntimeError(f"continuation terminal checkpoint hash mismatch: {row['case_id']}")


def run_training_task(manifest: str | Path, task_index: int, *, repository_root: str | Path = ".", admission_probe: bool = False, run_dir_override: str | Path | None = None) -> Path:
    """Run one frozen G1/G2/continuation task without shell CSV parsing."""
    row = load_training_task(manifest, task_index, repository_root=repository_root)
    run_dir = Path(run_dir_override).resolve() if run_dir_override is not None else Path(row["_run_dir"])
    if run_dir.exists():
        raise FileExistsError(f"task run directory already exists: {run_dir}")
    from phc2d.forward.full_cell import run_full_cell_forward

    result = run_full_cell_forward(row["_config_path"], model_name="global_local_band_operator_v1", seed=int(row["seed"]), run_dir=run_dir, admission_probe=admission_probe)
    audit_training_task_artifacts({**row, "_run_dir": str(result)}, admission_probe=admission_probe)
    return result


def audit_training_manifest(manifest: str | Path, start_index: int, end_index: int, *, repository_root: str | Path = ".") -> int:
    """Fail closed unless each selected frozen task is terminally auditable."""
    for task_index in range(int(start_index), int(end_index) + 1):
        audit_training_task_artifacts(load_training_task(manifest, task_index, repository_root=repository_root), admission_probe=False)
    return int(end_index) - int(start_index) + 1
