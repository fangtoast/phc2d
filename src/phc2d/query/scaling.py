"""Guarded query-form data-scaling runner.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from phc2d.data import inspect_query_dataset
from phc2d.query.runner import run_query_benchmark
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("query data-scaling config must be a mapping")
    return dict(payload)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return dict(value)


def _experiment_id(raw: Mapping[str, Any]) -> str:
    experiment = _as_mapping(raw.get("experiment"))
    return str(experiment.get("experiment_id", "query_data_scaling_v2"))


def _dataset_path(raw: Mapping[str, Any]) -> Path:
    dataset = _as_mapping(raw.get("dataset"))
    return Path(str(dataset.get("path", "data/query_dataset_v2_pilot_64_mpb.npz")))


def _scale_dataset_paths(raw: Mapping[str, Any]) -> dict[int, Path]:
    dataset = _as_mapping(raw.get("dataset"))
    raw_paths = dataset.get("paths_by_scale") or {}
    if not isinstance(raw_paths, Mapping):
        raise TypeError("dataset.paths_by_scale must be a mapping from scale to dataset path")
    return {int(scale): Path(str(path)) for scale, path in raw_paths.items()}


def _scale_sizes(raw: Mapping[str, Any]) -> list[int]:
    scaling = _as_mapping(raw.get("scaling"))
    values = scaling.get("sizes", [64])
    if not isinstance(values, list):
        values = [values]
    return [int(value) for value in values]


def _selected_scale_sizes(raw: Mapping[str, Any]) -> set[int] | None:
    scaling = _as_mapping(raw.get("scaling"))
    values = scaling.get("only_sizes", scaling.get("selected_sizes"))
    if values is None:
        return None
    if not isinstance(values, list):
        values = [values]
    return {int(value) for value in values}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        columns = ["status"]
        rows = [{"status": "empty"}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _available_n_from_audit(audit: Mapping[str, Any]) -> int:
    if audit.get("n_structures") is not None:
        return int(audit.get("n_structures", 0))
    available_by_scale = audit.get("available_by_scale")
    if not isinstance(available_by_scale, Mapping):
        return 0
    values = []
    for item in available_by_scale.values():
        if isinstance(item, Mapping) and item.get("n_structures") is not None:
            values.append(int(item.get("n_structures", 0)))
    return max(values) if values else 0


def _child_config(raw: Mapping[str, Any], *, scale: int, run_dir: Path, dataset_path: str | Path | None = None) -> dict[str, Any]:
    child = deepcopy(dict(raw))
    child.pop("scaling", None)
    child["experiment"] = {
        **_as_mapping(child.get("experiment")),
        "experiment_id": f"{_experiment_id(raw)}_N{scale}",
    }
    dataset = _as_mapping(child.get("dataset"))
    dataset.pop("paths_by_scale", None)
    if dataset_path is not None:
        dataset["path"] = str(dataset_path)
    child["dataset"] = dataset
    output = _as_mapping(child.get("output"))
    output.setdefault("runs_dir", "runs")
    output.pop("run_dir", None)
    output["write_query_report"] = False
    output["write_dataset_audit_report"] = False
    child["output"] = output
    child["parent_scaling_run_dir"] = str(run_dir)
    return child


def build_scale_plan(raw: Mapping[str, Any], audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Resolve which requested scales can be executed from the available dataset."""

    scaling = _as_mapping(raw.get("scaling"))
    require_exact = bool(scaling.get("require_exact_dataset_size", True))
    selected_sizes = _selected_scale_sizes(raw)
    available_by_scale = audit.get("available_by_scale")
    if isinstance(available_by_scale, Mapping):
        rows: list[dict[str, Any]] = []
        for size in _scale_sizes(raw):
            if selected_sizes is not None and int(size) not in selected_sizes:
                rows.append(
                    {
                        "scale_n": int(size),
                        "available_n": 0,
                        "dataset_path": "",
                        "status": "skipped_by_scale_filter",
                        "reason": f"N={size} is not selected by scaling.only_sizes",
                    }
                )
                continue
            scale_audit = available_by_scale.get(str(size)) or available_by_scale.get(size)
            if not isinstance(scale_audit, Mapping):
                rows.append(
                    {
                        "scale_n": int(size),
                        "available_n": 0,
                        "dataset_path": "",
                        "status": "pending_larger_v2_dataset",
                        "reason": f"no guarded V2 dataset path is configured for N={size}",
                    }
                )
                continue
            dataset_path = str(scale_audit.get("dataset_path", ""))
            if scale_audit.get("missing"):
                rows.append(
                    {
                        "scale_n": int(size),
                        "available_n": 0,
                        "dataset_path": dataset_path,
                        "status": "pending_larger_v2_dataset",
                        "reason": f"configured guarded V2 dataset for N={size} is missing: {dataset_path}",
                    }
                )
                continue
            available_n = int(scale_audit.get("n_structures", 0))
            if available_n == size:
                status = "runnable"
                reason = ""
            elif require_exact:
                status = "pending_larger_v2_dataset" if size > available_n else "pending_exact_v2_subset"
                reason = f"requested N={size} but configured guarded V2 dataset has N={available_n}: {dataset_path}"
            elif size < available_n:
                status = "runnable_subset"
                reason = ""
            else:
                status = "pending_larger_v2_dataset"
                reason = f"requested N={size} exceeds configured guarded V2 dataset N={available_n}: {dataset_path}"
            rows.append(
                {
                    "scale_n": int(size),
                    "available_n": available_n,
                    "dataset_path": dataset_path,
                    "status": status,
                    "reason": reason,
                }
            )
        return rows

    available_n = int(audit.get("n_structures", 0))
    rows: list[dict[str, Any]] = []
    for size in _scale_sizes(raw):
        if selected_sizes is not None and int(size) not in selected_sizes:
            rows.append(
                {
                    "scale_n": int(size),
                    "available_n": available_n,
                    "dataset_path": str(audit.get("dataset_path", _dataset_path(raw))),
                    "status": "skipped_by_scale_filter",
                    "reason": f"N={size} is not selected by scaling.only_sizes",
                }
            )
            continue
        if size == available_n:
            status = "runnable"
            reason = ""
        elif require_exact:
            status = "pending_larger_v2_dataset" if size > available_n else "pending_exact_v2_subset"
            reason = f"requested N={size} but only exact guarded V2 dataset N={available_n} is available"
        elif size < available_n:
            status = "runnable_subset"
            reason = ""
        else:
            status = "pending_larger_v2_dataset"
            reason = f"requested N={size} exceeds available guarded V2 dataset N={available_n}"
        rows.append(
            {
                "scale_n": int(size),
                "available_n": available_n,
                "dataset_path": str(audit.get("dataset_path", _dataset_path(raw))),
                "status": status,
                "reason": reason,
            }
        )
    return rows


def run_query_data_scaling(config_path: str | Path) -> Path:
    """Run the nearest executable guarded V2 query data-scaling gate."""

    started = time.perf_counter()
    path = Path(config_path)
    raw = _read_yaml(path)
    output = _as_mapping(raw.get("output"))
    run_dir = make_run_dir(
        _experiment_id(raw),
        runs_dir=Path(str(output.get("runs_dir", "runs"))),
        run_dir=Path(str(output["run_dir"])) if output.get("run_dir") else None,
    )
    meta = {
        "script": "phc2d query data-scaling",
        "config_path": str(path),
        "experiment_id": _experiment_id(raw),
        "dataset_path": str(_dataset_path(raw)),
        "validation_boundary": "neural_surrogate_no_mpb_meep",
        "pid": os.getpid(),
        "start_timestamp": datetime.now().astimezone().isoformat(),
    }
    logger = setup_logging(log_file=run_dir / RUN_LOG_NAME, run_meta=meta)
    ensure_figures_dir(run_dir)
    write_resolved_config(run_dir, yaml.safe_dump(dict(raw), sort_keys=False, allow_unicode=True))

    scale_paths = _scale_dataset_paths(raw)
    if scale_paths:
        audit = {
            "dataset_path": str(_dataset_path(raw)),
            "available_by_scale": {},
        }
        for scale, dataset_path in scale_paths.items():
            if dataset_path.is_file():
                audit["available_by_scale"][str(scale)] = inspect_query_dataset(dataset_path)
            else:
                audit["available_by_scale"][str(scale)] = {"dataset_path": str(dataset_path), "missing": True}
    else:
        audit = inspect_query_dataset(_dataset_path(raw))
    _write_json(run_dir / "query_schema.json", audit)
    plan_rows = build_scale_plan(raw, audit)
    _write_rows_csv(run_dir / "scale_plan.csv", plan_rows)

    child_rows: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    child_config_dir = run_dir / "child_configs"
    child_config_dir.mkdir(parents=True, exist_ok=True)

    for row in plan_rows:
        scale = int(row["scale_n"])
        if not str(row["status"]).startswith("runnable"):
            child_rows.append(row)
            continue
        child = _child_config(raw, scale=scale, run_dir=run_dir, dataset_path=row.get("dataset_path") or None)
        child_config_path = child_config_dir / f"{_experiment_id(raw)}_N{scale}.yaml"
        child_config_path.write_text(yaml.safe_dump(child, sort_keys=False, allow_unicode=True), encoding="utf-8")
        logger.info("Running guarded V2 query scaling child scale N=%s", scale)
        child_run_dir = run_query_benchmark(child_config_path)
        metrics_path = child_run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        child_row = {
            **row,
            "status": "completed",
            "child_run_dir": str(child_run_dir),
            "child_config_path": str(child_config_path),
            "cases": metrics.get("cases"),
            "n_failures": metrics.get("n_failures"),
            "resolved_device": metrics.get("resolved_device"),
            "device_name": metrics.get("device_name"),
            "wall_time_total_s": metrics.get("wall_time_total_s"),
            "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        }
        child_rows.append(child_row)
        for item in _read_csv_rows(child_run_dir / "leaderboard.csv"):
            item["scale_n"] = str(scale)
            item["child_run_dir"] = str(child_run_dir)
            leaderboard_rows.append(item)

    completed = [row for row in child_rows if row.get("status") == "completed"]
    skipped = [row for row in child_rows if row.get("status") == "skipped_by_scale_filter"]
    pending = [row for row in child_rows if row.get("status") not in {"completed", "skipped_by_scale_filter"}]
    _write_rows_csv(run_dir / "history.csv", child_rows)
    _write_rows_csv(run_dir / "leaderboard.csv", leaderboard_rows)
    _write_rows_csv(run_dir / "pending_scales.csv", pending)

    metrics_payload = {
        "status": "ok" if completed else "blocked",
        "stage": "query_data_scaling",
        "available_n": _available_n_from_audit(audit),
        "requested_scales": _scale_sizes(raw),
        "completed_scales": [int(row["scale_n"]) for row in completed],
        "skipped_scales": [int(row["scale_n"]) for row in skipped],
        "pending_scales": [int(row["scale_n"]) for row in pending],
        "wall_time_total_s": float(time.perf_counter() - started),
        "child_run_dirs": [str(row.get("child_run_dir")) for row in completed],
    }
    write_metrics(run_dir, metrics_payload)
    write_results(
        run_dir,
        {
            "status": metrics_payload["status"],
            "stage": "query_data_scaling",
            "run_dir": str(run_dir),
            "completed_scales": metrics_payload["completed_scales"],
            "skipped_scales": metrics_payload["skipped_scales"],
            "pending_scales": metrics_payload["pending_scales"],
            "neural_surrogate_no_mpb_meep": True,
        },
    )
    write_run_meta(
        run_dir,
        {
            **meta,
            "end_timestamp": datetime.now().astimezone().isoformat(),
            "wall_time_total_s": metrics_payload["wall_time_total_s"],
            "completed_scales": metrics_payload["completed_scales"],
            "skipped_scales": metrics_payload["skipped_scales"],
            "pending_scales": metrics_payload["pending_scales"],
        },
    )
    logger.info("Query data scaling finished: %s", run_dir)
    return run_dir


__all__ = ["build_scale_plan", "run_query_data_scaling"]
