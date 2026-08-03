"""Inverse-design diagnostic ledger and gate checker.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
import math
import time
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.runs import (
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_metrics,
    write_resolved_config,
    write_results,
)


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"YAML config not found: {source}")
    return _as_mapping(yaml.safe_load(source.read_text(encoding="utf-8")) or {}, name=str(source))


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {}
    return _as_mapping(json.loads(source.read_text(encoding="utf-8")), name=str(source))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})
    return out


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _to_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _path_from_run(run_dir: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else run_dir / path


def _source_config_from_run_meta(run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / "run.meta.json")
    config_path = meta.get("config_path")
    if not config_path:
        return {}
    source = Path(str(config_path))
    if not source.is_absolute():
        source = Path.cwd() / source
    if not source.is_file():
        return {}
    return _read_yaml(source)


def _experiment_id(run_dir: Path, fallback: str) -> str:
    results = _read_json(run_dir / "results.json")
    meta = _read_json(run_dir / "run.meta.json")
    return str(results.get("experiment_id") or meta.get("experiment_id") or run_dir.name or fallback)


def _evaluate_min(value: float | None, threshold: float | None) -> tuple[bool, str | None]:
    if threshold is None:
        return True, None
    if value is None:
        return False, f"missing >= {threshold}"
    return value >= threshold, None if value >= threshold else f"{value} < {threshold}"


def _evaluate_max(value: float | None, threshold: float | None) -> tuple[bool, str | None]:
    if threshold is None:
        return True, None
    if value is None:
        return False, f"missing <= {threshold}"
    return value <= threshold, None if value <= threshold else f"{value} > {threshold}"


def _candidate_gate(sample_dir: Path, thresholds: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    metrics = _read_json(sample_dir / "metrics.json")
    failures: list[str] = []

    validity = _to_float(metrics.get("validity_rate"))
    unique = _to_float(metrics.get("unique_rate"))
    p10 = _to_float(metrics.get("nearest_neighbor_hamming_p10"))
    best = _to_float(metrics.get("best_of_k_surrogate_mre_bands_2_10"))

    checks = [
        ("validity_rate", *_evaluate_min(validity, _to_float(thresholds.get("validity_rate_min")))),
        ("unique_rate", *_evaluate_min(unique, _to_float(thresholds.get("unique_rate_min")))),
        (
            "nearest_neighbor_hamming_p10",
            *_evaluate_min(p10, _to_float(thresholds.get("nearest_neighbor_hamming_p10_min"))),
        ),
        (
            "best_of_k_surrogate_mre_bands_2_10",
            *_evaluate_max(best, _to_float(thresholds.get("best_of_k_surrogate_mre_bands_2_10_max"))),
        ),
    ]
    for key, passed, reason in checks:
        if not passed:
            failures.append(f"{key}: {reason}")

    row = {
        "row_id": "candidate_preflight",
        "gate": "candidate_health",
        "evidence_type": "surrogate_side_sample",
        "source_run_dir": str(sample_dir),
        "experiment_id": _experiment_id(sample_dir, "inverse_sample"),
        "validation_boundary": "surrogate_side_no_mpb_meep",
        "diagnostic_only": False,
        "preflight_override": False,
        "gate_pass": not failures,
        "gate_status": "pass" if not failures else "fail",
        "failure_reasons": "; ".join(failures),
        "targets": _to_float(metrics.get("targets")),
        "target_seed_evaluations": _to_float(metrics.get("target_seed_evaluations")),
        "candidate_dim": _to_float(metrics.get("candidate_dim")),
        "validity_rate": validity,
        "validity_rate_min": _to_float(thresholds.get("validity_rate_min")),
        "unique_rate": unique,
        "unique_rate_min": _to_float(thresholds.get("unique_rate_min")),
        "nearest_neighbor_hamming_p10": p10,
        "nearest_neighbor_hamming_p10_min": _to_float(thresholds.get("nearest_neighbor_hamming_p10_min")),
        "nearest_neighbor_hamming_mean": _to_float(metrics.get("nearest_neighbor_hamming_mean")),
        "duplicate_cluster_count": _to_float(metrics.get("duplicate_cluster_count")),
        "largest_duplicate_cluster_fraction": _to_float(metrics.get("largest_duplicate_cluster_fraction")),
        "best_of_k_surrogate_mre_bands_2_10": best,
        "best_of_k_surrogate_mre_bands_2_10_max": _to_float(
            thresholds.get("best_of_k_surrogate_mre_bands_2_10_max")
        ),
        "surrogate_replay_gap_bands_2_10_pct": None,
        "best_of_k_replay_mre_bands_2_10_pct": None,
        "gap_f1": None,
        "mpb_replay_success_rate": None,
        "replay_backend": None,
    }
    return row, failures


def _replay_gate(replay_dir: Path, thresholds: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    metrics = _read_json(replay_dir / "metrics.json")
    source_config = _source_config_from_run_meta(replay_dir)
    replay_cfg = _as_mapping(source_config.get("replay"), name="source replay config")
    failures: list[str] = []

    backend = str(metrics.get("replay_backend", replay_cfg.get("backend", "")))
    success = _to_float(metrics.get("mpb_replay_success_rate"))
    diagnostic_only = _to_bool(replay_cfg.get("diagnostic_only"), default=False)
    preflight_override = _to_bool(replay_cfg.get("preflight_override"), default=False)

    if _to_bool(thresholds.get("require_real_mpb"), default=True) and backend != "real_mpb":
        failures.append(f"replay_backend: {backend} != real_mpb")
    min_success = _to_float(thresholds.get("mpb_replay_success_rate_min"), 1.0)
    success_pass, success_reason = _evaluate_min(success, min_success)
    if not success_pass:
        failures.append(f"mpb_replay_success_rate: {success_reason}")
    if _to_bool(thresholds.get("require_not_diagnostic_for_pass"), default=True):
        if diagnostic_only:
            failures.append("diagnostic_only replay cannot promote stage gate")
        if preflight_override:
            failures.append("preflight_override replay cannot promote stage gate")

    status = "diagnostic_only" if diagnostic_only or preflight_override else ("pass" if not failures else "fail")
    row = {
        "row_id": "real_mpb_replay",
        "gate": "replay_validation",
        "evidence_type": "real_mpb_replay" if backend == "real_mpb" else "non_physical_or_proxy_replay",
        "source_run_dir": str(replay_dir),
        "experiment_id": _experiment_id(replay_dir, "inverse_replay"),
        "validation_boundary": metrics.get("validation_boundary"),
        "diagnostic_only": diagnostic_only,
        "preflight_override": preflight_override,
        "gate_pass": not failures,
        "gate_status": status,
        "failure_reasons": "; ".join(failures),
        "targets": _to_float(metrics.get("targets")),
        "target_seed_evaluations": None,
        "candidate_dim": None,
        "validity_rate": None,
        "validity_rate_min": None,
        "unique_rate": None,
        "unique_rate_min": None,
        "nearest_neighbor_hamming_p10": None,
        "nearest_neighbor_hamming_p10_min": None,
        "nearest_neighbor_hamming_mean": None,
        "duplicate_cluster_count": None,
        "largest_duplicate_cluster_fraction": None,
        "best_of_k_surrogate_mre_bands_2_10": None,
        "best_of_k_surrogate_mre_bands_2_10_max": None,
        "best_of_k_surrogate_mre_bands_2_10_pct": _to_float(
            metrics.get("best_of_k_surrogate_mre_bands_2_10_pct")
        ),
        "best_of_k_replay_mre_bands_2_10_pct": _to_float(metrics.get("best_of_k_replay_mre_bands_2_10_pct")),
        "surrogate_replay_gap_bands_2_10_pct": _to_float(metrics.get("surrogate_replay_gap_bands_2_10_pct")),
        "gap_f1": _to_float(metrics.get("gap_f1")),
        "gap_precision": _to_float(metrics.get("gap_precision")),
        "gap_recall": _to_float(metrics.get("gap_recall")),
        "mpb_replay_success_rate": success,
        "replay_backend": backend,
    }
    return row, failures


def _candidate_failures(sample_dir: Path, thresholds: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
    rows = _read_csv(sample_dir / "ranked_candidates.csv")
    p10_threshold = _to_float(thresholds.get("nearest_neighbor_hamming_p10_min"))
    unique_threshold = _to_float(thresholds.get("unique_rate_min"))
    best_threshold = _to_float(thresholds.get("best_of_k_surrogate_mre_bands_2_10_max"))
    out: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []
        p10 = _to_float(row.get("nearest_neighbor_hamming_p10"))
        unique = _to_float(row.get("unique_rate"))
        best = _to_float(row.get("best_of_k_surrogate_mre_bands_2_10"))
        if p10_threshold is not None and (p10 is None or p10 < p10_threshold):
            reasons.append("nearest_neighbor_hamming_p10_below_threshold")
        if unique_threshold is not None and (unique is None or unique < unique_threshold):
            reasons.append("unique_rate_below_threshold")
        if best_threshold is not None and (best is None or best > best_threshold):
            reasons.append("surrogate_mre_above_threshold")
        if reasons:
            out.append(
                {
                    "source": "candidate_preflight",
                    "target_id": row.get("target_id"),
                    "target_index": row.get("target_index"),
                    "sample_id": row.get("sample_id"),
                    "failure_reason": ";".join(reasons),
                    "unique_rate": unique,
                    "nearest_neighbor_hamming_p10": p10,
                    "best_of_k_surrogate_mre_bands_2_10": best,
                    "duplicate_cluster_count": _to_float(row.get("duplicate_cluster_count")),
                    "largest_duplicate_cluster_fraction": _to_float(row.get("largest_duplicate_cluster_fraction")),
                }
            )
    out.sort(
        key=lambda item: (
            item.get("nearest_neighbor_hamming_p10") if item.get("nearest_neighbor_hamming_p10") is not None else -1.0,
            item.get("unique_rate") if item.get("unique_rate") is not None else -1.0,
            -(item.get("best_of_k_surrogate_mre_bands_2_10") or 0.0),
        )
    )
    return out[: max(int(limit), 0)]


def _replay_failures(replay_dir: Path, *, limit: int) -> list[dict[str, Any]]:
    rows = _read_csv(replay_dir / "per_target_replay_metrics.csv")
    failures: list[dict[str, Any]] = []
    for row in rows:
        replay_mre = _to_float(row.get("best_of_k_replay_mre_bands_2_10_pct"))
        gap = _to_float(row.get("surrogate_replay_gap_bands_2_10_pct"))
        failures.append(
            {
                "source": "real_mpb_replay",
                "target_id": row.get("target_id"),
                "target_index": row.get("target_index"),
                "sample_id": row.get("sample_id"),
                "failure_reason": "largest_replay_mre_or_surrogate_replay_gap",
                "best_of_k_replay_mre_bands_2_10_pct": replay_mre,
                "best_of_k_surrogate_mre_bands_2_10_pct": _to_float(
                    row.get("best_of_k_surrogate_mre_bands_2_10_pct")
                ),
                "surrogate_replay_gap_bands_2_10_pct": gap,
                "candidate_count_selected": _to_float(row.get("candidate_count_selected")),
                "selected_indices": row.get("selected_indices"),
            }
        )
    failures.sort(
        key=lambda item: (
            -(item.get("best_of_k_replay_mre_bands_2_10_pct") or 0.0),
            -(item.get("surrogate_replay_gap_bands_2_10_pct") or 0.0),
        )
    )
    return failures[: max(int(limit), 0)]


def run_inverse_diagnose(config_path: str | Path) -> Path:
    """Build a reproducible inverse-design gate ledger from existing artifacts.

    Parameters
    ----------
    config_path:
        YAML config containing sample/replay run directories and gate thresholds.
    Returns
    -------
    pathlib.Path
        The newly created diagnostic run directory.
    """

    wall_start = time.perf_counter()
    source_path = Path(config_path)
    raw = _read_yaml(source_path)
    experiment_cfg = _as_mapping(raw.get("experiment"), name="experiment")
    sample_cfg = _as_mapping(raw.get("sample"), name="sample")
    replay_cfg = _as_mapping(raw.get("replay"), name="replay")
    gates_cfg = _as_mapping(raw.get("gates"), name="gates")
    failure_cfg = _as_mapping(raw.get("failure_cases"), name="failure_cases")
    output_cfg = _as_mapping(raw.get("output"), name="output")

    experiment_id = str(experiment_cfg.get("experiment_id", "inverse_diagnostic_ledger"))
    run_dir = make_run_dir(
        experiment_id,
        runs_dir=output_cfg.get("runs_dir", "runs"),
        run_dir=output_cfg.get("run_dir"),
    )
    logger = setup_logging(
        log_file=run_dir / "run.log",
        run_meta={
            "script": "phc2d inverse diagnose",
            "config_path": str(source_path),
            "validation_boundary": "artifact_diagnostic_no_training_no_solver",
        },
    )
    logger.info("Building inverse diagnostic ledger from existing artifacts.")
    write_resolved_config(run_dir, yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(run_dir)

    sample_dir = Path(str(sample_cfg.get("run_dir", "")))
    replay_dir = Path(str(replay_cfg.get("run_dir", ""))) if replay_cfg.get("run_dir") else None
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample.run_dir does not exist: {sample_dir}")

    candidate_thresholds = _as_mapping(gates_cfg.get("candidate"), name="gates.candidate")
    replay_thresholds = _as_mapping(gates_cfg.get("replay"), name="gates.replay")

    ledger: list[dict[str, Any]] = []
    sample_row, sample_failures = _candidate_gate(sample_dir, candidate_thresholds)
    ledger.append(sample_row)

    replay_failures: list[str] = []
    replay_row: dict[str, Any] | None = None
    if replay_dir is not None:
        if not replay_dir.is_dir():
            raise FileNotFoundError(f"replay.run_dir does not exist: {replay_dir}")
        replay_row, replay_failures = _replay_gate(replay_dir, replay_thresholds)
        ledger.append(replay_row)

    overall_failures = list(sample_failures) + list(replay_failures)
    if replay_row is None:
        overall_failures.append("missing replay run")
    overall_pass = not overall_failures
    ledger.append(
        {
            "row_id": "overall_stage_gate",
            "gate": "stage_a_promotion",
            "evidence_type": "combined_artifact_diagnostic",
            "source_run_dir": f"{sample_dir}; {replay_dir}" if replay_dir else str(sample_dir),
            "experiment_id": experiment_id,
            "validation_boundary": "artifact_diagnostic_no_training_no_solver",
            "diagnostic_only": bool(replay_row.get("diagnostic_only")) if replay_row else False,
            "preflight_override": bool(replay_row.get("preflight_override")) if replay_row else False,
            "gate_pass": overall_pass,
            "gate_status": "pass" if overall_pass else "fail",
            "failure_reasons": "; ".join(overall_failures),
            "targets": sample_row.get("targets"),
            "candidate_dim": sample_row.get("candidate_dim"),
            "validity_rate": sample_row.get("validity_rate"),
            "unique_rate": sample_row.get("unique_rate"),
            "nearest_neighbor_hamming_p10": sample_row.get("nearest_neighbor_hamming_p10"),
            "best_of_k_surrogate_mre_bands_2_10": sample_row.get("best_of_k_surrogate_mre_bands_2_10"),
            "best_of_k_replay_mre_bands_2_10_pct": replay_row.get("best_of_k_replay_mre_bands_2_10_pct")
            if replay_row
            else None,
            "surrogate_replay_gap_bands_2_10_pct": replay_row.get("surrogate_replay_gap_bands_2_10_pct")
            if replay_row
            else None,
            "gap_f1": replay_row.get("gap_f1") if replay_row else None,
            "mpb_replay_success_rate": replay_row.get("mpb_replay_success_rate") if replay_row else None,
            "replay_backend": replay_row.get("replay_backend") if replay_row else None,
        }
    )

    candidate_limit = int(failure_cfg.get("candidate_max_rows", failure_cfg.get("max_rows", 25)))
    replay_limit = int(failure_cfg.get("replay_max_rows", failure_cfg.get("max_rows", 25)))
    failure_cases = _candidate_failures(sample_dir, candidate_thresholds, limit=candidate_limit)
    if replay_dir is not None:
        failure_cases.extend(_replay_failures(replay_dir, limit=replay_limit))

    _write_csv(run_dir / "inverse_gate_ledger.csv", ledger)
    _write_csv(run_dir / "failure_cases.csv", failure_cases)
    _write_csv(run_dir / "history.csv", ledger)
    (run_dir / "inverse_gate_ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = float(time.perf_counter() - wall_start)
    metrics = {
        "overall_gate_pass": float(1.0 if overall_pass else 0.0),
        "candidate_gate_pass": float(1.0 if sample_row["gate_pass"] else 0.0),
        "replay_gate_pass": float(1.0 if replay_row and replay_row["gate_pass"] else 0.0),
        "diagnostic_only": float(1.0 if replay_row and replay_row.get("diagnostic_only") else 0.0),
        "preflight_override": float(1.0 if replay_row and replay_row.get("preflight_override") else 0.0),
        "validity_rate": sample_row.get("validity_rate"),
        "unique_rate": sample_row.get("unique_rate"),
        "nearest_neighbor_hamming_p10": sample_row.get("nearest_neighbor_hamming_p10"),
        "best_of_k_surrogate_mre_bands_2_10": sample_row.get("best_of_k_surrogate_mre_bands_2_10"),
        "best_of_k_replay_mre_bands_2_10_pct": replay_row.get("best_of_k_replay_mre_bands_2_10_pct")
        if replay_row
        else None,
        "surrogate_replay_gap_bands_2_10_pct": replay_row.get("surrogate_replay_gap_bands_2_10_pct")
        if replay_row
        else None,
        "gap_f1": replay_row.get("gap_f1") if replay_row else None,
        "failure_case_count": len(failure_cases),
        "wall_time_total_s": elapsed,
    }
    results = {
        "action": "inverse diagnose",
        "stage": "artifact_diagnostic_no_training_no_solver",
        "run_dir": str(run_dir),
        "experiment_id": experiment_id,
        "sample_run_dir": str(sample_dir),
        "replay_run_dir": str(replay_dir) if replay_dir else None,
        "ledger_path": "inverse_gate_ledger.csv",
        "ledger_json_path": "inverse_gate_ledger.json",
        "failure_cases_path": "failure_cases.csv",
        "overall_gate_pass": overall_pass,
        "decision": "stage_gate_pass" if overall_pass else "stage_gate_fail",
        "failure_reasons": overall_failures,
        "note": "Diagnostic-only replays and preflight overrides never promote the stage gate.",
    }
    write_metrics(run_dir, metrics)
    write_results(run_dir, results)
    logger.info("Inverse diagnostic ledger written to %s", run_dir)
    return run_dir


def _optional_existing_dir(value: Any, *, name: str) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(str(value))
    if not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _extract_seed(target: Mapping[str, Any]) -> int | None:
    target_id = str(target.get("target_id", ""))
    marker = "_seed"
    if marker in target_id:
        tail = target_id.split(marker, 1)[1]
        digits = ""
        for char in tail:
            if char.isdigit() or (char == "-" and not digits):
                digits += char
            else:
                break
        if digits:
            try:
                return int(digits)
            except ValueError:
                return None
    for key in ("candidate_seed", "seed", "generator_seed"):
        value = target.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    metrics = _as_mapping(target.get("surrogate_metrics"), name="surrogate_metrics")
    value = metrics.get("candidate_seed")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _json_compact(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def _load_dataset_distribution(dataset_path: Path | None, *, threshold: float) -> dict[str, Any]:
    if dataset_path is None or not dataset_path.is_file():
        return {"available": False, "reason": "dataset path missing"}
    try:
        from phc2d.data import load_forward_dataset

        bundle = load_forward_dataset(dataset_path)
    except Exception as exc:  # pragma: no cover - defensive diagnostic downgrade.
        return {"available": False, "reason": f"failed to load dataset: {exc}"}

    train_idx = np.asarray(bundle.idx_train, dtype=np.int64)
    rho = np.asarray(bundle.rho, dtype=np.float32).reshape(int(bundle.rho.shape[0]), -1)
    if train_idx.size == 0:
        return {"available": False, "reason": "empty train split"}
    train = rho[train_idx]
    fill = np.mean(train >= float(threshold), axis=1)
    return {
        "available": True,
        "dataset_path": str(dataset_path),
        "train_count": int(train.shape[0]),
        "candidate_dim": int(train.shape[1]),
        "feature_min": np.min(train, axis=0).astype(float),
        "feature_max": np.max(train, axis=0).astype(float),
        "fill_min": float(np.min(fill)),
        "fill_p01": float(np.percentile(fill, 1.0)),
        "fill_p99": float(np.percentile(fill, 99.0)),
        "fill_max": float(np.max(fill)),
    }


def _distribution_flags(candidate: np.ndarray, distribution: Mapping[str, Any], *, threshold: float) -> dict[str, Any]:
    if not distribution.get("available"):
        return {
            "training_distribution_available": False,
            "feature_ood_count": None,
            "feature_ood_fraction": None,
            "fill_fraction": float(np.mean(candidate.reshape(-1) >= float(threshold))),
            "fill_ood": None,
        }
    feature_min = np.asarray(distribution["feature_min"], dtype=np.float32)
    feature_max = np.asarray(distribution["feature_max"], dtype=np.float32)
    flat = np.asarray(candidate, dtype=np.float32).reshape(-1)
    dim = min(int(flat.shape[0]), int(feature_min.shape[0]), int(feature_max.shape[0]))
    if dim == 0:
        feature_ood = np.asarray([], dtype=bool)
    else:
        feature_ood = (flat[:dim] < feature_min[:dim]) | (flat[:dim] > feature_max[:dim])
    fill_fraction_value = float(np.mean(flat >= float(threshold))) if flat.size else 0.0
    fill_ood = fill_fraction_value < float(distribution["fill_p01"]) or fill_fraction_value > float(
        distribution["fill_p99"]
    )
    return {
        "training_distribution_available": True,
        "feature_ood_count": int(np.sum(feature_ood)),
        "feature_ood_fraction": float(np.mean(feature_ood)) if feature_ood.size else 0.0,
        "fill_fraction": fill_fraction_value,
        "fill_ood": bool(fill_ood),
    }


def _selected_errors(target: Mapping[str, Any]) -> list[float | None]:
    metrics = _as_mapping(target.get("surrogate_metrics"), name="surrogate_metrics")
    raw = metrics.get("selected_reconstruction_errors", target.get("selected_reconstruction_errors", []))
    if not isinstance(raw, list):
        return []
    out: list[float | None] = []
    for item in raw:
        out.append(_to_float(item))
    return out


def _selected_predictions(target: Mapping[str, Any]) -> list[Any]:
    raw = target.get("selected_surrogate_predictions")
    if raw is None:
        raw = _as_mapping(target.get("surrogate_metrics"), name="surrogate_metrics").get("selected_surrogate_predictions")
    return raw if isinstance(raw, list) else []


def _target_condition(target: Mapping[str, Any], condition_target: str) -> Any:
    condition = _as_mapping(target.get("condition"), name="condition")
    return condition.get(condition_target, condition.get("Wh"))


def _build_design_diagnostic_tables(
    *,
    targets: list[Mapping[str, Any]],
    condition_target: str,
    design_bounds: tuple[float, float],
    threshold: float,
    distribution: Mapping[str, Any],
    include_design_variables: bool,
    include_target_output: bool,
    include_prediction_output: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    lo, hi = design_bounds
    for target in targets:
        target_id = str(target.get("target_id", "unknown"))
        target_index = int(_to_float(target.get("target_index"), 0.0) or 0)
        seed = _extract_seed(target)
        candidates = np.asarray(target.get("candidates", []), dtype=np.float32)
        if candidates.ndim < 2 or int(candidates.shape[0]) == 0:
            continue
        selected_indices = target.get("selected_indices", [])
        if not isinstance(selected_indices, list):
            selected_indices = []
        errors = _selected_errors(target)
        predictions = _selected_predictions(target)
        target_output = _target_condition(target, condition_target)
        metrics = _as_mapping(target.get("surrogate_metrics"), name="surrogate_metrics")
        filtered = _as_mapping(target.get("filtered_metrics"), name="filtered_metrics")
        best_error = None

        for selected_rank, raw_index in enumerate(selected_indices):
            try:
                candidate_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if candidate_index < 0 or candidate_index >= int(candidates.shape[0]):
                continue
            candidate = np.asarray(candidates[candidate_index], dtype=np.float32)
            flat = candidate.reshape(-1)
            below = flat < float(lo)
            above = flat > float(hi)
            bound_violations = int(np.sum(below | above))
            dist = _distribution_flags(candidate, distribution, threshold=threshold)
            error = errors[selected_rank] if selected_rank < len(errors) else None
            if error is not None:
                best_error = error if best_error is None else min(float(best_error), float(error))
            predicted_output = predictions[selected_rank] if selected_rank < len(predictions) else None
            row: dict[str, Any] = {
                "target_id": target_id,
                "target_index": target_index,
                "seed": seed,
                "selected_rank": int(selected_rank),
                "candidate_index": candidate_index,
                "selected_surrogate_mre": error,
                "prediction_available": predicted_output is not None,
                "design_dim": int(flat.shape[0]),
                "design_min": float(np.min(flat)) if flat.size else None,
                "design_max": float(np.max(flat)) if flat.size else None,
                "design_mean": float(np.mean(flat)) if flat.size else None,
                "design_std": float(np.std(flat)) if flat.size else None,
                "lower_bound": float(lo),
                "upper_bound": float(hi),
                "bound_violation_count": bound_violations,
                "bound_violation_fraction": float(bound_violations / max(int(flat.shape[0]), 1)),
                **dist,
            }
            if include_design_variables:
                row["design_variables_json"] = _json_compact(flat.tolist())
            if include_target_output:
                row["target_output_json"] = _json_compact(target_output)
            if include_prediction_output:
                row["predicted_output_json"] = _json_compact(predicted_output) if predicted_output is not None else ""
            selected_rows.append(row)

        target_rows.append(
            {
                "target_id": target_id,
                "target_index": target_index,
                "seed": seed,
                "candidate_count": int(candidates.shape[0]),
                "selected_count": int(len(selected_indices)),
                "best_selected_surrogate_mre": best_error,
                "unique_rate": _to_float(filtered.get("unique_rate")),
                "nearest_neighbor_hamming_p10": _to_float(filtered.get("nearest_neighbor_hamming_p10")),
                "nearest_neighbor_hamming_mean": _to_float(filtered.get("nearest_neighbor_hamming_mean")),
                "duplicate_cluster_count": _to_float(filtered.get("duplicate_cluster_count")),
                "largest_duplicate_cluster_fraction": _to_float(filtered.get("largest_duplicate_cluster_fraction")),
                "best_of_k_surrogate_mre": _to_float(metrics.get("best_of_k_surrogate_mre")),
                "best_of_k_surrogate_mre_bands_2_10": _to_float(metrics.get("best_of_k_surrogate_mre_bands_2_10")),
                "cross_surrogate_disagreement_bands_2_10": _to_float(
                    metrics.get("cross_surrogate_disagreement_bands_2_10")
                ),
                "gap_descriptor_error": _to_float(metrics.get("gap_descriptor_error")),
            }
        )
    multi_rows = _multi_start_rows(selected_rows)
    return selected_rows, target_rows, multi_rows


def _multi_start_rows(selected_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in selected_rows:
        grouped.setdefault(int(row["target_index"]), []).append(row)
    out: list[dict[str, Any]] = []
    for target_index, rows in sorted(grouped.items()):
        seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") is not None})
        scored = [row for row in rows if row.get("selected_surrogate_mre") is not None]
        best = min(scored, key=lambda row: float(row["selected_surrogate_mre"])) if scored else None
        out.append(
            {
                "target_index": int(target_index),
                "seed_count": int(len(seeds)),
                "seeds": _json_compact(seeds),
                "selected_candidate_count": int(len(rows)),
                "best_seed": int(best["seed"]) if best is not None and best.get("seed") is not None else None,
                "best_candidate_index": int(best["candidate_index"]) if best is not None else None,
                "best_selected_surrogate_mre": float(best["selected_surrogate_mre"]) if best is not None else None,
                "all_selected_bound_clean": bool(
                    all(int(row.get("bound_violation_count") or 0) == 0 for row in rows)
                ),
                "any_fill_ood": bool(any(bool(row.get("fill_ood")) for row in rows if row.get("fill_ood") is not None)),
            }
        )
    return out


def _copy_objective_history(objective_dir: Path | None, run_dir: Path) -> tuple[list[dict[str, str]], str | None]:
    if objective_dir is None:
        return [], None
    source = objective_dir / "history.csv"
    if not source.is_file():
        return [], None
    rows = _read_csv(source)
    shutil.copyfile(source, run_dir / "optimization_history.csv")
    return rows, "optimization_history.csv"


def _write_design_report(
    path: Path,
    *,
    sample_dir: Path,
    objective_dir: Path | None,
    replay_dir: Path | None,
    metrics: Mapping[str, Any],
    results: Mapping[str, Any],
) -> None:
    lines = [
        "<!--",
        "  Inverse design diagnostic report.",
        "  Author: fangtoast",
        "  Email: fangtoast@foxmail.com",
        "  Copyright (c) 2026 fangtoast. All rights reserved.",
        "-->",
        "",
        "# Inverse Design Diagnostic Report",
        "",
        "## Evidence Boundary",
        "",
        "- This report audits existing inverse artifacts; it does not train a model or run MPB/Meep.",
        f"- Sample run: `{sample_dir}`",
        f"- Objective-train run: `{objective_dir}`" if objective_dir else "- Objective-train run: `not provided`",
        f"- Replay run: `{replay_dir}`" if replay_dir else "- Replay run: `not provided`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "targets",
        "selected_candidate_count",
        "n_seeds_detected",
        "multi_start_target_count",
        "design_bound_violation_rate",
        "selected_design_feature_ood_rate",
        "selected_design_fill_ood_rate",
        "prediction_available_rate",
        "objective_history_rows",
        "real_replay_available",
        "diagnostic_only",
        "preflight_override",
        "best_selected_surrogate_mre_mean",
        "best_selected_surrogate_mre_min",
    ):
        lines.append(f"| `{key}` | `{metrics.get(key)}` |")
    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            f"- `selected_candidate_trace.csv`: {results.get('selected_candidate_trace_path')}",
            f"- `target_summary.csv`: {results.get('target_summary_path')}",
            f"- `multi_start_summary.csv`: {results.get('multi_start_summary_path')}",
            f"- `optimization_history.csv`: {results.get('optimization_history_path')}",
            "",
            "## Trust Notes",
            "",
            "- `prediction_available_rate=0` means the source sample artifact did not store selected surrogate predictions.",
            "- Surrogate-side target errors are not physical validation.",
            "- A diagnostic or preflight-override replay cannot promote the stage gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_inverse_design_diagnostic(config_path: str | Path) -> Path:
    """Generate an inverse-design trace report from existing train/sample/replay artifacts."""

    wall_start = time.perf_counter()
    source_path = Path(config_path)
    raw = _read_yaml(source_path)
    experiment_cfg = _as_mapping(raw.get("experiment"), name="experiment")
    sample_cfg = _as_mapping(raw.get("sample"), name="sample")
    objective_cfg = _as_mapping(raw.get("objective_train"), name="objective_train")
    replay_cfg = _as_mapping(raw.get("replay"), name="replay")
    dataset_cfg = _as_mapping(raw.get("dataset"), name="dataset")
    diagnostic_cfg = _as_mapping(raw.get("diagnostics"), name="diagnostics")
    output_cfg = _as_mapping(raw.get("output"), name="output")

    experiment_id = str(experiment_cfg.get("experiment_id", "inverse_design_diagnostic"))
    sample_dir = _optional_existing_dir(sample_cfg.get("run_dir"), name="sample.run_dir")
    if sample_dir is None:
        raise ValueError("sample.run_dir is required")
    objective_dir = _optional_existing_dir(objective_cfg.get("run_dir"), name="objective_train.run_dir")
    replay_dir = _optional_existing_dir(replay_cfg.get("run_dir"), name="replay.run_dir")

    run_dir = make_run_dir(
        experiment_id,
        runs_dir=output_cfg.get("runs_dir", "runs"),
        run_dir=output_cfg.get("run_dir"),
    )
    logger = setup_logging(
        log_file=run_dir / "run.log",
        run_meta={
            "script": "phc2d inverse design-diagnose",
            "config_path": str(source_path),
            "validation_boundary": "inverse_design_artifact_diagnostic_no_training_no_solver",
        },
    )
    logger.info("Building inverse-design trace diagnostic from existing artifacts.")
    write_resolved_config(run_dir, yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(run_dir)

    sample_resolved = _read_yaml(sample_dir / "config.resolved.yaml")
    sample_inverse = _as_mapping(sample_resolved.get("inverse"), name="sample inverse")
    sample_dataset = _as_mapping(sample_resolved.get("dataset"), name="sample dataset")
    condition_target = str(dataset_cfg.get("condition_target", sample_dataset.get("condition_target", "Wh")))
    threshold = float(diagnostic_cfg.get("threshold", sample_inverse.get("threshold", 0.5)))
    raw_bounds = diagnostic_cfg.get("design_bounds", sample_inverse.get("candidate_value_range", [0.0, 1.0]))
    if not isinstance(raw_bounds, list | tuple) or len(raw_bounds) != 2:
        raise ValueError("diagnostics.design_bounds must contain exactly two values")
    design_bounds = (float(raw_bounds[0]), float(raw_bounds[1]))
    dataset_path = Path(str(dataset_cfg.get("path", sample_dataset.get("path", "")))) if dataset_cfg or sample_dataset else None
    distribution = (
        _load_dataset_distribution(dataset_path, threshold=threshold)
        if _to_bool(diagnostic_cfg.get("check_training_distribution"), default=True)
        else {"available": False, "reason": "disabled"}
    )

    payload = _read_json(sample_dir / "candidates.json")
    targets = [dict(item) for item in payload.get("targets", [])]
    max_targets = diagnostic_cfg.get("max_targets")
    if max_targets is not None:
        targets = targets[: max(int(max_targets), 0)]

    selected_rows, target_rows, multi_rows = _build_design_diagnostic_tables(
        targets=targets,
        condition_target=condition_target,
        design_bounds=design_bounds,
        threshold=threshold,
        distribution=distribution,
        include_design_variables=_to_bool(diagnostic_cfg.get("include_design_variables"), default=True),
        include_target_output=_to_bool(diagnostic_cfg.get("include_target_output"), default=True),
        include_prediction_output=_to_bool(diagnostic_cfg.get("include_prediction_output"), default=True),
    )
    objective_rows, objective_history_path = _copy_objective_history(objective_dir, run_dir)
    replay_metrics = _read_json(replay_dir / "metrics.json") if replay_dir else {}
    replay_source_cfg = _source_config_from_run_meta(replay_dir) if replay_dir else {}
    replay_source = _as_mapping(replay_source_cfg.get("replay"), name="replay source") if replay_source_cfg else {}

    _write_csv(run_dir / "selected_candidate_trace.csv", selected_rows)
    _write_csv(run_dir / "target_summary.csv", target_rows)
    _write_csv(run_dir / "multi_start_summary.csv", multi_rows)
    _write_csv(
        run_dir / "history.csv",
        [
            {
                "stage": "inverse_design_diagnostic",
                "targets": int(len(target_rows)),
                "selected_candidate_count": int(len(selected_rows)),
                "objective_history_rows": int(len(objective_rows)),
            }
        ],
    )

    selected_errors = [
        float(row["selected_surrogate_mre"])
        for row in selected_rows
        if row.get("selected_surrogate_mre") is not None
    ]
    feature_ood_values = [
        float(row["feature_ood_fraction"])
        for row in selected_rows
        if row.get("feature_ood_fraction") is not None
    ]
    fill_ood_values = [1.0 if row.get("fill_ood") else 0.0 for row in selected_rows if row.get("fill_ood") is not None]
    prediction_flags = [1.0 if row.get("prediction_available") else 0.0 for row in selected_rows]
    bound_flags = [1.0 if int(row.get("bound_violation_count") or 0) > 0 else 0.0 for row in selected_rows]
    seeds = sorted({int(row["seed"]) for row in selected_rows if row.get("seed") is not None})
    elapsed = float(time.perf_counter() - wall_start)
    metrics = {
        "targets": int(len(target_rows)),
        "selected_candidate_count": int(len(selected_rows)),
        "n_seeds_detected": int(len(seeds)),
        "multi_start_target_count": int(sum(1 for row in multi_rows if int(row.get("seed_count") or 0) > 1)),
        "design_bound_violation_rate": float(np.mean(bound_flags)) if bound_flags else 0.0,
        "selected_design_feature_ood_rate": float(np.mean([value > 0.0 for value in feature_ood_values]))
        if feature_ood_values
        else 0.0,
        "selected_design_fill_ood_rate": float(np.mean(fill_ood_values)) if fill_ood_values else 0.0,
        "prediction_available_rate": float(np.mean(prediction_flags)) if prediction_flags else 0.0,
        "objective_history_rows": int(len(objective_rows)),
        "real_replay_available": float(1.0 if replay_dir and replay_metrics.get("replay_backend") == "real_mpb" else 0.0),
        "diagnostic_only": float(1.0 if _to_bool(replay_source.get("diagnostic_only"), default=False) else 0.0),
        "preflight_override": float(1.0 if _to_bool(replay_source.get("preflight_override"), default=False) else 0.0),
        "best_selected_surrogate_mre_mean": float(np.mean(selected_errors)) if selected_errors else None,
        "best_selected_surrogate_mre_min": float(np.min(selected_errors)) if selected_errors else None,
        "best_of_k_replay_mre_bands_2_10_pct": _to_float(replay_metrics.get("best_of_k_replay_mre_bands_2_10_pct")),
        "surrogate_replay_gap_bands_2_10_pct": _to_float(replay_metrics.get("surrogate_replay_gap_bands_2_10_pct")),
        "gap_f1": _to_float(replay_metrics.get("gap_f1")),
        "wall_time_total_s": elapsed,
    }
    results = {
        "action": "inverse design-diagnose",
        "stage": "inverse_design_artifact_diagnostic_no_training_no_solver",
        "run_dir": str(run_dir),
        "experiment_id": experiment_id,
        "sample_run_dir": str(sample_dir),
        "objective_train_run_dir": str(objective_dir) if objective_dir else None,
        "replay_run_dir": str(replay_dir) if replay_dir else None,
        "selected_candidate_trace_path": "selected_candidate_trace.csv",
        "target_summary_path": "target_summary.csv",
        "multi_start_summary_path": "multi_start_summary.csv",
        "optimization_history_path": objective_history_path,
        "report_path": "inverse_design_report.md",
        "prediction_output_note": (
            "selected surrogate predictions were present"
            if metrics["prediction_available_rate"]
            else "source sample artifact did not store selected surrogate predictions"
        ),
        "trust_boundary": "surrogate trace unless replay metrics are explicitly present",
    }
    _write_design_report(
        run_dir / "inverse_design_report.md",
        sample_dir=sample_dir,
        objective_dir=objective_dir,
        replay_dir=replay_dir,
        metrics=metrics,
        results=results,
    )
    write_metrics(run_dir, metrics)
    write_results(run_dir, results)
    logger.info("Inverse-design trace diagnostic written to %s", run_dir)
    return run_dir


__all__ = ["run_inverse_design_diagnostic", "run_inverse_diagnose"]
