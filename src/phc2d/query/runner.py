"""Native query-form band benchmark runner.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.data import (
    BandQueryDataset,
    build_physical_query_masks,
    inspect_query_dataset,
    load_forward_dataset,
    build_query_masks,
    recover_k_path,
    split_indices,
    validate_query_split_integrity,
)
from phc2d.models import (
    architecture_fingerprint,
    build_model,
    describe_model_identity,
    get_model_spec,
)
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
from phc2d.training import (
    DeviceConfig,
    autocast_context,
    collect_device_metadata,
    create_grad_scaler,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    resolve_device,
    synchronize_device,
)


VERIFIED_K_PATH_STATUSES = frozenset(
    {
        "physically_verified",
        "physically_verified_with_documented_numerical_tolerance",
    }
)


def _is_verified_k_path_status(value: str | None) -> bool:
    return str(value or "") in VERIFIED_K_PATH_STATUSES


def _require_torch() -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Query benchmark requires optional dependency 'torch'.") from exc
    return torch, DataLoader


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("query config must be a mapping")
    return dict(payload)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return dict(value)


def _as_list(value: Any, *, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if not isinstance(value, list):
        return [value]
    return list(value)


def _experiment_id(raw: Mapping[str, Any], fallback: str) -> str:
    experiment = _as_mapping(raw.get("experiment"))
    return str(experiment.get("experiment_id", fallback))


def _dataset_path(raw: Mapping[str, Any]) -> Path:
    dataset = _as_mapping(raw.get("dataset"))
    return Path(str(dataset.get("path", "data/tiny.npz")))


def _runtime(raw: Mapping[str, Any]) -> DeviceConfig:
    training = _as_mapping(raw.get("training"))
    return DeviceConfig.from_mapping(_as_mapping(raw.get("runtime")), fallback_device=str(training.get("device", "cpu")))


def _seeds(raw: Mapping[str, Any]) -> list[int]:
    training = _as_mapping(raw.get("training"))
    value = training.get("seeds", training.get("seed", [0]))
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(value)]


def _learning_rates(raw: Mapping[str, Any]) -> list[float]:
    training = _as_mapping(raw.get("training"))
    value = training.get("learning_rates")
    if value is None:
        value = training.get("learning_rate", training.get("lr", 1.0e-3))
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _learning_rate_label(value: float) -> str:
    return f"lr{float(value):.0e}".replace("+", "").replace("-", "m")


def _model_names(raw: Mapping[str, Any]) -> list[str]:
    model_cfg = _as_mapping(raw.get("model"))
    return [str(v) for v in _as_list(model_cfg.get("candidates", model_cfg.get("models")), default=["mlp_query"])]


def _model_params(raw: Mapping[str, Any], model_name: str | None = None) -> dict[str, Any]:
    model_cfg = _as_mapping(raw.get("model"))
    params = {}
    params.update(_as_mapping(model_cfg.get("params")))
    params.update(_as_mapping(model_cfg.get("overrides")))
    if model_name is not None:
        params.update(_as_mapping(_as_mapping(model_cfg.get("params_by_model")).get(model_name)))
    return params


def _protocol_names(raw: Mapping[str, Any]) -> list[str]:
    values = _as_list(raw.get("protocols"), default=["unseen_structure_all_k"])
    names = []
    for item in values:
        if isinstance(item, Mapping):
            names.append(str(item.get("name")))
        else:
            names.append(str(item))
    return names


def _protocol_lookup(protocols: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve query protocols by both registry key and reported protocol name."""

    lookup = dict(protocols)
    for protocol in protocols.values():
        reported_name = str(protocol.name)
        existing = lookup.get(reported_name)
        if existing is not None and existing != protocol:
            raise ValueError(f"Conflicting query protocol name: {reported_name}")
        lookup[reported_name] = protocol
    return lookup


def _limits(raw: Mapping[str, Any]) -> dict[str, int | None]:
    cfg = _as_mapping(raw.get("limits"))
    return {
        "max_train_structures": None if cfg.get("max_train_structures") is None else int(cfg.get("max_train_structures")),
        "max_eval_structures": None if cfg.get("max_eval_structures") is None else int(cfg.get("max_eval_structures")),
        "max_train_steps": None if cfg.get("max_train_steps") is None else int(cfg.get("max_train_steps")),
        "max_eval_queries": None if cfg.get("max_eval_queries") is None else int(cfg.get("max_eval_queries")),
    }


def _query_cfg(raw: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _as_mapping(raw.get("dataset"))
    query = _as_mapping(dataset.get("query"))
    query.update(_as_mapping(raw.get("query")))
    return query


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_optimizer(raw: Mapping[str, Any], model: Any, *, lr: float, torch: Any) -> Any:
    training = _as_mapping(raw.get("training"))
    name = str(training.get("optimizer", "adam")).lower()
    weight_decay = float(training.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=weight_decay)
    raise ValueError(f"Unsupported query optimizer: {name!r}")


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
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _make_run(raw: Mapping[str, Any], config_path: Path, *, suffix: str) -> tuple[Path, Any, dict[str, Any]]:
    output = _as_mapping(raw.get("output"))
    run_dir = make_run_dir(
        _experiment_id(raw, suffix),
        runs_dir=Path(str(output.get("runs_dir", "runs"))),
        run_dir=Path(str(output["run_dir"])) if output.get("run_dir") else None,
    )
    meta = {
        "script": f"phc2d query {suffix}",
        "config_path": str(config_path),
        "experiment_id": _experiment_id(raw, suffix),
        "dataset_path": str(_dataset_path(raw)),
        "validation_boundary": "neural_surrogate_no_mpb_meep",
        "pid": os.getpid(),
        "start_timestamp": datetime.now().astimezone().isoformat(),
    }
    logger = setup_logging(log_file=run_dir / RUN_LOG_NAME, run_meta=meta)
    ensure_figures_dir(run_dir)
    write_resolved_config(run_dir, yaml.safe_dump(dict(raw), sort_keys=False, allow_unicode=True))
    return run_dir, logger, meta


def _finalize(run_dir: Path, meta: Mapping[str, Any], started: float, extra: Mapping[str, Any] | None = None) -> None:
    payload = dict(meta)
    payload.update(dict(extra or {}))
    payload["end_timestamp"] = datetime.now().astimezone().isoformat()
    payload["wall_time_total_s"] = float(time.perf_counter() - started)
    write_run_meta(run_dir, payload)


def _write_audit_artifacts(run_dir: Path, dataset_path: Path) -> dict[str, Any]:
    bundle = load_forward_dataset(dataset_path)
    k_path = recover_k_path(bundle)
    protocols = build_physical_query_masks(k_path) if k_path.is_physical else build_query_masks(bundle.nk, has_segment_metadata=k_path.has_segment_metadata)
    audit = inspect_query_dataset(dataset_path)
    split_manifest = {
        "idx_train": [int(v) for v in bundle.idx_train],
        "idx_val": [int(v) for v in bundle.idx_val],
        "idx_test": [int(v) for v in bundle.idx_test],
        "protocols": {
            name: {
                "actual_name": protocol.name,
                "train_k_indices": list(protocol.train_k_indices),
                "eval_k_indices": list(protocol.eval_k_indices),
                "holdout_kind": protocol.holdout_kind,
            }
            for name, protocol in protocols.items()
        },
    }
    k_payload = {
        "coordinates": k_path.coordinates.tolist(),
        "coordinate_source": k_path.source,
        "is_physical": k_path.is_physical,
        "dimension": k_path.dimension,
        "has_segment_metadata": k_path.has_segment_metadata,
        "repeated_endpoint_indices": list(k_path.repeated_endpoint_indices),
        "nodes": list(k_path.nodes),
        "path_s": None if k_path.path_s is None else k_path.path_s.tolist(),
        "segment_id": None if k_path.segment_id is None else k_path.segment_id.tolist(),
        "segment_local_t": None if k_path.segment_local_t is None else k_path.segment_local_t.tolist(),
        "physical_k_id": list(k_path.physical_k_id),
        "k_path_id": k_path.k_path_id,
        "k_path_hash": k_path.k_path_hash,
        "verification_status": k_path.verification_status,
    }
    _write_json(run_dir / "query_schema.json", audit)
    _write_json(run_dir / "k_path.json", k_payload)
    _write_json(run_dir / "split_manifest.json", split_manifest)
    _write_json(run_dir / "integrity_summary.json", validate_query_split_integrity(bundle, protocols))
    return audit


def _write_dataset_audit_report(run_dir: Path, audit: Mapping[str, Any]) -> Path:
    report = Path("docs/results/query_dataset_audit_20260618.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    k_path = _as_mapping(audit.get("k_path"))
    lines = [
        "# Query Dataset And k-path Audit",
        "",
        "日期：2026-06-18",
        f"run_dir: `{run_dir}`",
        f"dataset: `{audit.get('dataset_path')}`",
        "",
        "## Schema",
        "",
        f"- number of structures: `{audit.get('n_structures')}`",
        f"- Wh shape: `{audit.get('Wh_shape')}`",
        f"- Wh dtype: `{audit.get('Wh_dtype')}`",
        f"- Nk / Nb: `{audit.get('Nk')}` / `{audit.get('Nb')}`",
        f"- splits: `{audit.get('splits')}`",
        "",
        "## k coordinate source",
        "",
        f"- coordinate_source: `{k_path.get('coordinate_source')}`",
        f"- is_physical: `{k_path.get('is_physical')}`",
        f"- dimension: `{k_path.get('dimension')}`",
        f"- has_segment_metadata: `{k_path.get('has_segment_metadata')}`",
        f"- verification_status: `{k_path.get('verification_status')}`",
        f"- k_path_id: `{k_path.get('k_path_id')}`",
        "",
    ]
    if bool(k_path.get("is_physical")):
        lines.append(
            "结论：当前 dataset 已发现物理 k-path sidecar；若 `verification_status` 不是 `physically_verified`，仍只能作为 candidate physical-k metadata，Formal Full-2A 保持 blocked。"
        )
    else:
        lines.append(
            "结论：当前 NPZ 未保存可追溯的物理 `k_points` 或 Γ/X/M segment metadata；只能形成 `index-query` 边界，不能声称完成连续物理 k-space interpolation/extrapolation。"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_query_audit(config_path: str | Path) -> Path:
    started = time.perf_counter()
    path = Path(config_path)
    raw = _read_yaml(path)
    raw = {**raw, "experiment": {**_as_mapping(raw.get("experiment")), "experiment_id": _experiment_id(raw, "query_dataset_audit")}}
    run_dir, logger, meta = _make_run(raw, path, suffix="audit")
    audit = _write_audit_artifacts(run_dir, _dataset_path(raw))
    report = None
    if bool(_as_mapping(raw.get("output")).get("write_dataset_audit_report", True)):
        report = _write_dataset_audit_report(run_dir, audit)
    metrics = {
        "status": "ok",
        "n_structures": int(audit["n_structures"]),
        "Nk": int(audit["Nk"]),
        "Nb": int(audit["Nb"]),
        "physical_k_coordinates_available": bool(_as_mapping(audit["k_path"]).get("is_physical")),
    }
    write_metrics(run_dir, metrics)
    write_results(run_dir, {"status": "ok", "stage": "query_dataset_audit", "report": None if report is None else str(report), "run_dir": str(run_dir)})
    logger.info("Query dataset audit finished: %s", run_dir)
    _finalize(run_dir, meta, started, metrics)
    return run_dir


def _collate(batch: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    return {
        "structure": torch.as_tensor(np.stack([row["structure"] for row in batch]), dtype=torch.float32),
        "k_coordinate": torch.as_tensor(np.stack([row["k_coordinate"] for row in batch]), dtype=torch.float32),
        "target": torch.as_tensor(np.stack([row["target"] for row in batch]), dtype=torch.float32),
        "k_index": torch.as_tensor(np.stack([row["k_index"] for row in batch]), dtype=torch.long),
        "sample_id": [str(row["sample_id"]) for row in batch],
        "query_id": [str(row["query_id"]) for row in batch],
    }


def _materialize_query_dataset(dataset: BandQueryDataset, torch: Any) -> dict[str, Any]:
    """Materialize one protocol split with one structure transform per sample."""

    sample_indices = np.asarray(dataset.sample_indices, dtype=np.int64)
    k_indices = np.asarray(dataset.k_indices, dtype=np.int64)
    structures_by_sample = np.stack(
        [dataset[local_pos * len(k_indices)]["structure"] for local_pos in range(len(sample_indices))],
        axis=0,
    ).astype(np.float32)
    structures = np.repeat(structures_by_sample, len(k_indices), axis=0)
    k_coordinate = np.tile(np.asarray(dataset.k_path.coordinates[k_indices], dtype=np.float32), (len(sample_indices), 1))
    targets = np.asarray(dataset.bundle.Wh[np.ix_(sample_indices, k_indices)], dtype=np.float32).reshape(
        len(sample_indices) * len(k_indices),
        int(dataset.bundle.nb),
    )
    k_index = np.tile(k_indices, len(sample_indices)).astype(np.int64)
    sample_ids: list[str] = []
    query_ids: list[str] = []
    for sample_index in sample_indices:
        metadata = dataset.bundle.metadata[int(sample_index)] if int(sample_index) < len(dataset.bundle.metadata) else {}
        sample_id = dataset._sample_id(int(sample_index), metadata)
        for k_value in k_indices:
            sample_ids.append(sample_id)
            query_ids.append(f"{sample_id}:{dataset.k_path.k_path_id or 'kpath_unknown'}:k_{int(k_value):04d}")
    return {
        "structure": torch.as_tensor(structures, dtype=torch.float32),
        "k_coordinate": torch.as_tensor(k_coordinate, dtype=torch.float32),
        "target": torch.as_tensor(targets, dtype=torch.float32),
        "k_index": torch.as_tensor(k_index, dtype=torch.long),
        "sample_id": sample_ids,
        "query_id": query_ids,
    }


def _take_tensor_batch(data: Mapping[str, Any], indices: Any, *, device: Any, device_info: Any) -> tuple[Any, Any, Any]:
    structure = data["structure"].index_select(0, indices).to(device, non_blocking=device_info.non_blocking_transfer)
    k_coordinate = data["k_coordinate"].index_select(0, indices).to(device, non_blocking=device_info.non_blocking_transfer)
    target = data["target"].index_select(0, indices).to(device, non_blocking=device_info.non_blocking_transfer)
    return structure, k_coordinate, target


def _take_indices(values: np.ndarray, limit: int | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if limit is not None:
        arr = arr[: int(limit)]
    return arr


def _query_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, elapsed_s: float, parameter_count: int) -> dict[str, Any]:
    err = y_pred - y_true
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(y_true), 1.0e-8)
    ordering = np.diff(y_pred, axis=-1) < 0.0 if y_pred.shape[-1] > 1 else np.zeros_like(y_pred, dtype=bool)
    return {
        "query_mae": mae,
        "query_rmse": float(math.sqrt(mse)),
        "query_mre_pct": float(100.0 * np.mean(np.abs(err) / denom)),
        "query_mre_bands_2_10_pct": float(100.0 * np.mean(np.abs(err[..., 1:]) / denom[..., 1:])) if y_true.shape[-1] > 1 else None,
        "ordering_violation_rate": float(np.mean(ordering)) if ordering.size else 0.0,
        "eval_time_s": float(elapsed_s),
        "query_inference_time_per_query_ms": float(1000.0 * elapsed_s / max(int(y_true.shape[0]), 1)),
        "throughput_queries_per_s": float(int(y_true.shape[0]) / elapsed_s) if elapsed_s > 0 else 0.0,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(parameter_count),
    }


def _query_per_band_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    case_id: str,
    protocol: str,
    model: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if y_true.ndim < 2 or y_true.shape[-1] == 0:
        return rows
    err = y_pred - y_true
    for band_idx in range(int(y_true.shape[-1])):
        band_true = y_true[..., band_idx]
        band_pred = y_pred[..., band_idx]
        band_err = err[..., band_idx]
        denom = np.maximum(np.abs(band_true), 1.0e-8)
        rows.append(
            {
                "case_id": case_id,
                "protocol": protocol,
                "model": model,
                "seed": int(seed),
                "band_index": int(band_idx),
                "band_one_based": int(band_idx + 1),
                "n_eval_queries": int(band_true.size),
                "query_mae": float(np.mean(np.abs(band_err))),
                "query_rmse": float(math.sqrt(float(np.mean(np.square(band_err))))),
                "query_mre_pct": float(100.0 * np.mean(np.abs(band_err) / denom)),
            }
        )
    return rows


def _evaluate(model: Any, loader: Any, torch: Any, device: Any, device_info: Any, *, max_queries: int | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int]]:
    true_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    k_indices: list[int] = []
    seen = 0
    model.eval()
    synchronize_device(device, torch_module=torch)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            structure = batch["structure"].to(device, non_blocking=device_info.non_blocking_transfer)
            k_coordinate = batch["k_coordinate"].to(device, non_blocking=device_info.non_blocking_transfer)
            target = batch["target"].to(device, non_blocking=device_info.non_blocking_transfer)
            with autocast_context(device_info, device, torch_module=torch):
                pred = model(structure, k_coordinate)
            true_rows.append(target.detach().cpu().numpy())
            pred_rows.append(pred.detach().cpu().numpy())
            k_indices.extend([int(v) for v in batch["k_index"].detach().cpu().numpy().reshape(-1)])
            seen += int(target.shape[0])
            if max_queries is not None and seen >= int(max_queries):
                break
    synchronize_device(device, torch_module=torch)
    elapsed = time.perf_counter() - start
    y_true = np.concatenate(true_rows, axis=0)
    y_pred = np.concatenate(pred_rows, axis=0)
    if max_queries is not None:
        y_true = y_true[: int(max_queries)]
        y_pred = y_pred[: int(max_queries)]
        k_indices = k_indices[: int(max_queries)]
    metrics = _query_metrics(y_true, y_pred, elapsed_s=elapsed, parameter_count=sum(int(p.numel()) for p in model.parameters()))
    return metrics, y_true, y_pred, k_indices


def _evaluate_materialized(
    model: Any,
    data: Mapping[str, Any],
    torch: Any,
    device: Any,
    device_info: Any,
    *,
    batch_size: int,
    max_queries: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int]]:
    n_total = int(data["target"].shape[0])
    n_eval = n_total if max_queries is None else min(n_total, int(max_queries))
    true_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    model.eval()
    synchronize_device(device, torch_module=torch)
    start = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(0, n_eval, max(1, int(batch_size))):
            idx = torch.arange(
                batch_start,
                min(batch_start + max(1, int(batch_size)), n_eval),
                dtype=torch.long,
            )
            structure, k_coordinate, target = _take_tensor_batch(data, idx, device=device, device_info=device_info)
            with autocast_context(device_info, device, torch_module=torch):
                pred = model(structure, k_coordinate)
            true_rows.append(target.detach().cpu().numpy())
            pred_rows.append(pred.detach().cpu().numpy())
    synchronize_device(device, torch_module=torch)
    elapsed = time.perf_counter() - start
    y_true = np.concatenate(true_rows, axis=0)
    y_pred = np.concatenate(pred_rows, axis=0)
    k_indices = [int(v) for v in data["k_index"][:n_eval].detach().cpu().numpy().reshape(-1)]
    metrics = _query_metrics(
        y_true,
        y_pred,
        elapsed_s=elapsed,
        parameter_count=sum(int(p.numel()) for p in model.parameters()),
    )
    return metrics, y_true, y_pred, k_indices


def _query_case_checkpoint(run_dir: Path, case_id: str) -> Path:
    return run_dir / "cases" / f"{case_id}.pt"


def _latest_query_epoch_checkpoint(run_dir: Path, case_id: str) -> tuple[int, Path] | None:
    """Return the highest numbered epoch checkpoint for a query case."""

    best: tuple[int, Path] | None = None
    prefix = f"{case_id}_checkpoint_epoch_"
    for path in (run_dir / "cases").glob(f"{case_id}_checkpoint_epoch_*.pt"):
        if not path.stem.startswith(prefix):
            continue
        try:
            epoch = int(path.stem.removeprefix(prefix))
        except ValueError:
            continue
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return best


def _mean_std_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate only comparable rows; learning rates are never mixed."""

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["protocol"]), str(row["model"]), float(row["learning_rate"])), []).append(row)
    output: list[dict[str, Any]] = []
    metric_names = [
        "query_mae",
        "query_rmse",
        "query_mre_pct",
        "query_mre_bands_2_10_pct",
        "ordering_violation_rate",
        "train_time_s",
        "query_inference_time_per_query_ms",
        "throughput_queries_per_s",
        "peak_gpu_memory_mb",
        "parameter_count",
    ]
    for (protocol, model, learning_rate), group in grouped.items():
        item: dict[str, Any] = {"protocol": protocol, "model": model, "learning_rate": learning_rate, "seeds": [int(row["seed"]) for row in group], "n_seeds": len(group)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    output.sort(key=lambda row: float(row.get("query_mre_bands_2_10_pct_mean", float("inf"))))
    return output


def _run_query_matrix(config_path: str | Path, *, mode: str) -> Path:
    started = time.perf_counter()
    path = Path(config_path)
    raw = _read_yaml(path)
    fair_selection = bool(_as_mapping(raw.get("fair_selection")).get("enabled", False))
    run_dir, logger, meta = _make_run(raw, path, suffix=mode)
    audit = _write_audit_artifacts(run_dir, _dataset_path(raw))
    output_cfg = _as_mapping(raw.get("output"))
    if bool(output_cfg.get("write_dataset_audit_report", True)):
        _write_dataset_audit_report(run_dir, audit)
    query_cfg = _query_cfg(raw)
    requires_physical = bool(query_cfg.get("requires_physical_k_coordinates", False))
    allow_index = bool(query_cfg.get("allow_index_query", False))
    k_path_info = _as_mapping(audit.get("k_path"))
    if requires_physical and not bool(k_path_info.get("is_physical")) and not allow_index:
        metrics = {
            "status": "blocked",
            "blocker": "physical_k_coordinates_unavailable",
            "physical_k_coordinates_available": False,
            "index_query_preflight_allowed": True,
        }
        write_metrics(run_dir, metrics)
        write_results(
            run_dir,
            {
                "status": "blocked",
                "stage": f"query_{mode}",
                "run_dir": str(run_dir),
                "reason": "Dataset lacks physical k_points / segment metadata; formal physical query benchmark was not started.",
            },
        )
        if bool(output_cfg.get("write_query_report", True)):
            _write_blocked_query_report(run_dir)
        _finalize(run_dir, meta, started, metrics)
        logger.warning("Query %s blocked by missing physical k coordinates: %s", mode, run_dir)
        return run_dir
    verification_status = str(k_path_info.get("verification_status") or "")
    if mode == "benchmark" and requires_physical and not _is_verified_k_path_status(verification_status):
        metrics = {
            "status": "blocked",
            "blocker": "physical_k_path_not_verified",
            "physical_k_coordinates_available": bool(k_path_info.get("is_physical")),
            "k_path_verified": False,
            "verification_status": verification_status or "missing",
        }
        write_metrics(run_dir, metrics)
        write_results(
            run_dir,
            {
                "status": "blocked",
                "stage": f"query_{mode}",
                "run_dir": str(run_dir),
                "reason": "Physical k-path sidecar is not physically_verified; Formal Full-2A was not started.",
            },
        )
        if bool(output_cfg.get("write_query_report", True)):
            _write_blocked_query_report(run_dir)
        _finalize(run_dir, meta, started, metrics)
        logger.warning("Query %s blocked by unverified physical k-path: %s", mode, run_dir)
        return run_dir

    torch, _DataLoader = _require_torch()
    runtime = _runtime(raw)
    device_info = resolve_device(runtime, torch_module=torch)
    device = device_info.torch_device
    reset_peak_gpu_memory(device, torch_module=torch)
    bundle = load_forward_dataset(_dataset_path(raw))
    if fair_selection:
        fair_cfg = _as_mapping(raw.get("fair_selection"))
        expected_hash = fair_cfg.get("expected_dataset_sha256")
        actual_hash = hashlib.sha256(_dataset_path(raw).read_bytes()).hexdigest()
        if expected_hash and str(expected_hash) != actual_hash:
            raise ValueError("Fair query benchmark dataset hash does not match the frozen protocol.")
        expected_sizes = _as_mapping(fair_cfg.get("expected_split_sizes"))
        actual_sizes = {"train": len(bundle.idx_train), "val": len(bundle.idx_val), "test": len(bundle.idx_test)}
        if expected_sizes and any(int(expected_sizes[name]) != actual_sizes[name] for name in actual_sizes):
            raise ValueError("Fair query benchmark split sizes do not match the frozen protocol.")
    k_path = recover_k_path(bundle)
    protocols = build_physical_query_masks(k_path) if k_path.is_physical else build_query_masks(bundle.nk, has_segment_metadata=k_path.has_segment_metadata)
    protocol_lookup = _protocol_lookup(protocols)
    names = _protocol_names(raw)
    limits = _limits(raw)
    training = _as_mapping(raw.get("training"))
    batch_size = max(1, int(training.get("batch_size", 8)))
    epochs = max(1, int(training.get("max_epochs", training.get("epochs", 1))))
    raw_checkpoint_epochs = training.get("checkpoint_epochs", ())
    if isinstance(raw_checkpoint_epochs, (list, tuple)):
        checkpoint_epochs = {int(item) for item in raw_checkpoint_epochs if int(item) > 0}
    elif raw_checkpoint_epochs:
        checkpoint_epochs = {int(raw_checkpoint_epochs)}
    else:
        checkpoint_epochs = set()
    learning_rates = _learning_rates(raw)
    seeds = _seeds(raw)
    case_rows: list[dict[str, Any]] = []
    per_k_rows: list[dict[str, Any]] = []
    per_band_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def _record_case(
        *,
        protocol_name: str,
        requested_protocol: str,
        model_name: str,
        base_case_id: str,
        case_id: str,
        seed: int,
        learning_rate: float,
        model: Any,
        structure_shape: tuple[int, ...],
        train_queries: int,
        eval_data: Mapping[str, Any],
        checkpoint: Path,
        train_time_s: float | None,
        gradient_finite: bool,
        write_checkpoint: bool,
        resumed_from_case_artifact: bool,
        resumed_from_epoch_checkpoint: bool,
        resumed_checkpoint_epoch: int | None,
        resumed_checkpoint_path: Path | None,
        evaluation_split: str = "test",
    ) -> None:
        metrics, y_true, y_pred, k_indices = _evaluate_materialized(
            model,
            eval_data,
            torch,
            device,
            device_info,
            batch_size=batch_size,
            max_queries=limits["max_eval_queries"],
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if write_checkpoint:
            torch.save(
                {
                    "case_id": case_id,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "learning_rate": float(learning_rate),
                    "train_time_s": train_time_s,
                    "seed": int(seed),
                    "protocol": protocol_name,
                    "resolved_config": raw,
                },
                checkpoint,
            )
        reloaded = build_model(
            model_name,
            input_shape=(1, *structure_shape),
            output_shape=(bundle.nb,),
            k_dim=k_path.dimension,
            **{k: v for k, v in _model_params(raw, model_name).items() if k in get_model_spec(model_name).defaults},
        ).to(device)
        reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
        row = {
            "case_id": case_id,
            "base_case_id": base_case_id,
            "protocol": protocol_name,
            "requested_protocol": requested_protocol,
            "model": model_name,
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "evaluation_split": evaluation_split,
            "train_time_s": None if train_time_s is None else float(train_time_s),
            "n_train_queries": int(train_queries),
            "n_eval_queries": int(y_true.shape[0]),
            "checkpoint_save_load_ok": True,
            "gradient_finite": bool(gradient_finite),
            "resolved_device": device_info.resolved_device,
            "k_coordinate_source": k_path.source,
            "is_physical_k_coordinate": k_path.is_physical,
            "peak_gpu_memory_mb": peak_gpu_memory_mb(device, torch_module=torch),
            "checkpoint_path": str(checkpoint),
            "resumed_from_case_artifact": bool(resumed_from_case_artifact),
            "resumed_from_epoch_checkpoint": bool(resumed_from_epoch_checkpoint),
            "resumed_checkpoint_epoch": resumed_checkpoint_epoch,
            "resumed_checkpoint_path": str(resumed_checkpoint_path) if resumed_checkpoint_path else None,
            **metrics,
        }
        case_rows.append(row)
        per_band_rows.extend(
            {
                **band_row,
                "base_case_id": base_case_id,
                "learning_rate": float(learning_rate),
                "evaluation_split": evaluation_split,
            }
            for band_row in _query_per_band_rows(
                y_true,
                y_pred,
                case_id=str(row["case_id"]),
                protocol=protocol_name,
                model=model_name,
                seed=int(seed),
            )
        )
        log_prefix = "Query case restored" if resumed_from_case_artifact else "Query case finished"
        logger.info(
            "%s: case_id=%s train_time_s=%s eval_time_s=%.3f bands_2_10_mre=%.6f",
            log_prefix,
            row["case_id"],
            "missing" if train_time_s is None else f"{float(train_time_s):.3f}",
            float(row.get("eval_time_s", 0.0)),
            float(row.get("query_mre_bands_2_10_pct", 0.0) or 0.0),
        )
        identity = describe_model_identity(
            model,
            model_name,
            model_kwargs={"input_shape": (1, *structure_shape), "output_shape": (bundle.nb,), "k_dim": k_path.dimension},
        )
        identity["case_id"] = row["case_id"]
        identity["base_case_id"] = base_case_id
        identity["learning_rate"] = float(learning_rate)
        identity["architecture_fingerprint"] = architecture_fingerprint(model)
        identity_rows.append(identity)
        k_array = np.asarray(k_indices)
        for k_index in sorted(set(k_indices)):
            mask = k_array == int(k_index)
            if not np.any(mask):
                continue
            km = _query_metrics(y_true[mask], y_pred[mask], elapsed_s=0.0, parameter_count=int(metrics["parameter_count"]))
            per_k_rows.append(
                {
                    "case_id": row["case_id"],
                    "base_case_id": base_case_id,
                    "protocol": protocol_name,
                    "model": model_name,
                    "seed": int(seed),
                    "learning_rate": float(learning_rate),
                    "evaluation_split": evaluation_split,
                    "k_index": int(k_index),
                    **km,
                }
            )

    for protocol_key in names:
        if protocol_key not in protocol_lookup:
            failures.append({"protocol": protocol_key, "error": "unknown_protocol"})
            continue
        protocol = protocol_lookup[protocol_key]
        train_ds = BandQueryDataset(
            bundle,
            sample_indices=_take_indices(split_indices(bundle, "train"), limits["max_train_structures"]),
            k_indices=protocol.train_k_indices,
            k_path=k_path,
            split="train",
            protocol=protocol.name,
        )
        selection_split = "val" if fair_selection else "test"
        eval_ds = BandQueryDataset(
            bundle,
            sample_indices=_take_indices(split_indices(bundle, selection_split), limits["max_eval_structures"]),
            k_indices=protocol.eval_k_indices,
            k_path=k_path,
            split="test",
            protocol=protocol.name,
        )
        first = train_ds[0]
        structure_shape = tuple(int(v) for v in np.asarray(first["structure"]).shape)
        materialize_start = time.perf_counter()
        train_data = _materialize_query_dataset(train_ds, torch)
        eval_data = _materialize_query_dataset(eval_ds, torch)
        logger.info(
            "Query protocol materialized: protocol=%s train_queries=%s eval_queries=%s elapsed_s=%.3f",
            protocol.name,
            len(train_ds),
            len(eval_ds),
            float(time.perf_counter() - materialize_start),
        )
        for model_name in _model_names(raw):
            spec = get_model_spec(model_name)
            if not spec.supports_query:
                failures.append({"protocol": protocol_key, "model": model_name, "error": "model_does_not_support_query"})
                continue
            for learning_rate in learning_rates:
                lr_case_suffix = _learning_rate_label(learning_rate)
                for seed in seeds:
                    base_case_id = f"{protocol.name}_{model_name}_seed{seed}"
                    case_id = base_case_id if len(learning_rates) == 1 else f"{base_case_id}_{lr_case_suffix}"
                    checkpoint = _query_case_checkpoint(run_dir, case_id)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    model = build_model(
                        model_name,
                        input_shape=(1, *structure_shape),
                        output_shape=(bundle.nb,),
                        k_dim=k_path.dimension,
                        **{k: v for k, v in _model_params(raw, model_name).items() if k in get_model_spec(model_name).defaults},
                    ).to(device)
                    if checkpoint.is_file():
                        payload = torch.load(checkpoint, map_location=device, weights_only=False)
                        model.load_state_dict(payload["state_dict"])
                        _record_case(
                            protocol_name=protocol.name,
                            requested_protocol=protocol_key,
                            model_name=model_name,
                            base_case_id=base_case_id,
                            case_id=case_id,
                            seed=int(seed),
                            learning_rate=float(learning_rate),
                            model=model,
                            structure_shape=structure_shape,
                            train_queries=len(train_ds),
                            eval_data=eval_data,
                            checkpoint=checkpoint,
                            train_time_s=payload.get("train_time_s"),
                            gradient_finite=True,
                            write_checkpoint=False,
                            resumed_from_case_artifact=True,
                            resumed_from_epoch_checkpoint=False,
                            resumed_checkpoint_epoch=None,
                            resumed_checkpoint_path=None,
                            evaluation_split=selection_split,
                        )
                        continue

                    optimizer = _make_optimizer(raw, model, lr=learning_rate, torch=torch)
                    loss_fn = torch.nn.MSELoss()
                    scaler = create_grad_scaler(device_info, device, torch_module=torch)
                    resume_checkpoint = _latest_query_epoch_checkpoint(run_dir, case_id)
                    resumed_from_epoch_checkpoint = False
                    resumed_checkpoint_epoch: int | None = None
                    resumed_checkpoint_path: Path | None = None
                    start_epoch = 1
                    if resume_checkpoint is not None:
                        checkpoint_epoch, checkpoint_path = resume_checkpoint
                        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
                        model.load_state_dict(payload["state_dict"])
                        resumed_from_epoch_checkpoint = True
                        resumed_checkpoint_epoch = int(payload.get("epoch", checkpoint_epoch))
                        resumed_checkpoint_path = checkpoint_path
                        start_epoch = min(int(resumed_checkpoint_epoch) + 1, epochs + 1)
                        logger.info(
                            "Resuming query case from %s at epoch %s: case_id=%s",
                            checkpoint_path,
                            resumed_checkpoint_epoch,
                            case_id,
                        )
                    train_start = time.perf_counter()
                    step_count = 0
                    model.train()
                    n_train = int(train_data["target"].shape[0])
                    for epoch in range(start_epoch, epochs + 1):
                        order = torch.randperm(n_train)
                        for batch_start in range(0, n_train, batch_size):
                            batch_indices = order[batch_start : batch_start + batch_size]
                            structure, k_coordinate, target = _take_tensor_batch(
                                train_data,
                                batch_indices,
                                device=device,
                                device_info=device_info,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            with autocast_context(device_info, device, torch_module=torch):
                                loss = loss_fn(model(structure, k_coordinate), target)
                            if bool(getattr(scaler, "is_enabled", lambda: False)()):
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                optimizer.step()
                            step_count += 1
                        if epoch in checkpoint_epochs:
                            epoch_checkpoint = run_dir / "cases" / f"{case_id}_checkpoint_epoch_{epoch}.pt"
                            epoch_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                {
                                    "model_name": model_name,
                                    "state_dict": model.state_dict(),
                                    "epoch": int(epoch),
                                    "learning_rate": float(learning_rate),
                                    "resolved_config": raw,
                                },
                                epoch_checkpoint,
                            )
                            if limits["max_train_steps"] is not None and step_count >= int(limits["max_train_steps"]):
                                break
                        if limits["max_train_steps"] is not None and step_count >= int(limits["max_train_steps"]):
                            break
                    synchronize_device(device, torch_module=torch)
                    train_time_s = time.perf_counter() - train_start
                    _record_case(
                        protocol_name=protocol.name,
                        requested_protocol=protocol_key,
                        model_name=model_name,
                        base_case_id=base_case_id,
                        case_id=case_id,
                        seed=int(seed),
                        learning_rate=float(learning_rate),
                        model=model,
                        structure_shape=structure_shape,
                        train_queries=len(train_ds),
                        eval_data=eval_data,
                        checkpoint=checkpoint,
                        train_time_s=float(train_time_s),
                        gradient_finite=all(param.grad is None or bool(torch.isfinite(param.grad).all().item()) for param in model.parameters()),
                        write_checkpoint=True,
                        resumed_from_case_artifact=False,
                        resumed_from_epoch_checkpoint=resumed_from_epoch_checkpoint,
                        resumed_checkpoint_epoch=resumed_checkpoint_epoch,
                        resumed_checkpoint_path=resumed_checkpoint_path,
                        evaluation_split=selection_split,
                    )

    validation_rows: list[dict[str, Any]] = []
    validation_per_band_rows: list[dict[str, Any]] = []
    validation_per_k_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    if fair_selection:
        validation_rows = list(case_rows)
        validation_per_band_rows = list(per_band_rows)
        validation_per_k_rows = list(per_k_rows)
        selected: dict[tuple[str, str], float] = {}
        for protocol_name in names:
            for model_name in _model_names(raw):
                candidates: list[tuple[float, float, list[dict[str, Any]]]] = []
                for learning_rate in learning_rates:
                    rows = [row for row in validation_rows if row["protocol"] == protocol_name and row["model"] == model_name and float(row["learning_rate"]) == float(learning_rate)]
                    if {int(row["seed"]) for row in rows} != set(seeds):
                        continue
                    candidates.append((float(np.mean([float(row["query_mre_bands_2_10_pct"]) for row in rows])), float(learning_rate), rows))
                if not candidates:
                    raise RuntimeError(f"No complete validation-selected learning rate for protocol={protocol_name} model={model_name}.")
                mean, learning_rate, rows = min(candidates, key=lambda item: item[0])
                selected[(protocol_name, model_name)] = learning_rate
                selection_rows.append({"protocol": protocol_name, "model": model_name, "selected_learning_rate": learning_rate, "selection_metric": "query_mre_bands_2_10_pct", "validation_mean": mean, "seed_count": len(rows), "selection_split": "val", "final_evaluation_split": "test", "checkpoint_paths": [row["checkpoint_path"] for row in rows]})

        case_rows.clear()
        per_band_rows.clear()
        per_k_rows.clear()
        identity_rows.clear()
        for protocol_key in names:
            protocol = protocol_lookup[protocol_key]
            train_ds = BandQueryDataset(bundle, sample_indices=_take_indices(split_indices(bundle, "train"), limits["max_train_structures"]), k_indices=protocol.train_k_indices, k_path=k_path, split="train", protocol=protocol.name)
            test_ds = BandQueryDataset(bundle, sample_indices=_take_indices(split_indices(bundle, "test"), limits["max_eval_structures"]), k_indices=protocol.eval_k_indices, k_path=k_path, split="test", protocol=protocol.name)
            test_data = _materialize_query_dataset(test_ds, torch)
            first = train_ds[0]
            structure_shape = tuple(int(v) for v in np.asarray(first["structure"]).shape)
            for model_name in _model_names(raw):
                learning_rate = selected[(protocol_key, model_name)]
                rows = [row for row in validation_rows if row["protocol"] == protocol.name and row["model"] == model_name and float(row["learning_rate"]) == learning_rate]
                for selected_row in rows:
                    checkpoint = Path(str(selected_row["checkpoint_path"]))
                    model = build_model(model_name, input_shape=(1, *structure_shape), output_shape=(bundle.nb,), k_dim=k_path.dimension, **{k: v for k, v in _model_params(raw, model_name).items() if k in get_model_spec(model_name).defaults}).to(device)
                    payload = torch.load(checkpoint, map_location=device, weights_only=False)
                    model.load_state_dict(payload["state_dict"])
                    _record_case(protocol_name=protocol.name, requested_protocol=protocol_key, model_name=model_name, base_case_id=str(selected_row["base_case_id"]), case_id=str(selected_row["case_id"]), seed=int(selected_row["seed"]), learning_rate=learning_rate, model=model, structure_shape=structure_shape, train_queries=len(train_ds), eval_data=test_data, checkpoint=checkpoint, train_time_s=selected_row.get("train_time_s"), gradient_finite=bool(selected_row.get("gradient_finite", True)), write_checkpoint=False, resumed_from_case_artifact=True, resumed_from_epoch_checkpoint=False, resumed_checkpoint_epoch=None, resumed_checkpoint_path=None, evaluation_split="test")
        if any(len([row for row in case_rows if row["protocol"] == selection["protocol"] and row["model"] == selection["model"]]) != len(seeds) for selection in selection_rows):
            raise RuntimeError("Fair query test aggregation does not have exactly three selected seeds.")

    leaderboard = _mean_std_rows(case_rows)
    _write_rows_csv(run_dir / "history.csv", case_rows)
    _write_rows_csv(run_dir / "leaderboard.csv", leaderboard)
    _write_rows_csv(run_dir / "per_k_metrics.csv", per_k_rows)
    _write_rows_csv(run_dir / "per_band_metrics.csv", per_band_rows)
    _write_rows_csv(run_dir / "continuity_metrics.csv", [])
    _write_rows_csv(run_dir / "protocol_metrics.csv", leaderboard)
    _write_rows_csv(run_dir / "full_wh_baseline_comparison.csv", [])
    _write_rows_csv(run_dir / "failure_cases.csv", failures)
    if fair_selection:
        _write_rows_csv(run_dir / "validation_history.csv", validation_rows)
        _write_rows_csv(run_dir / "validation_per_band_metrics.csv", validation_per_band_rows)
        _write_rows_csv(run_dir / "validation_per_k_metrics.csv", validation_per_k_rows)
        _write_json(run_dir / "validation_selection_manifest.json", {"protocol": "validation_selected_lr_v1", "selections": selection_rows, "test_driven_selection": False})
    _write_json(run_dir / "model_identity_summary.json", {"models": identity_rows})
    _write_json(run_dir / "results.json", {"status": "ok", "stage": f"query_{mode}", "case_rows": len(case_rows), "failures": failures})
    metrics_payload = {
        "status": "ok",
        "stage": f"query_{mode}",
        "cases": len(case_rows),
        "n_failures": len(failures),
        "physical_k_coordinates_available": bool(k_path.is_physical),
        "k_coordinate_source": k_path.source,
        "wall_time_total_s": float(time.perf_counter() - started),
        **collect_device_metadata(device_info, runtime),
    }
    metrics_payload["peak_gpu_memory_mb"] = peak_gpu_memory_mb(device, torch_module=torch)
    write_metrics(run_dir, metrics_payload)
    write_results(run_dir, {"status": "ok", "stage": f"query_{mode}", "run_dir": str(run_dir), "case_rows": len(case_rows), "neural_surrogate_no_mpb_meep": True})
    if bool(output_cfg.get("write_query_report", True)):
        _write_query_report(run_dir, leaderboard, audit, mode=mode)
    _finalize(run_dir, {**meta, **collect_device_metadata(device_info, runtime)}, started, metrics_payload)
    logger.info("Query %s finished: %s", mode, run_dir)
    return run_dir


def _write_blocked_query_report(run_dir: Path) -> Path:
    report = Path("docs/results/query_benchmark_gpu_20260618.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Formal GPU Full-2A Query-form Benchmark",
        "",
        "日期：2026-06-18",
        f"run_dir: `{run_dir}`",
        "",
        "## 状态",
        "",
        "Formal physical k-query benchmark 未启动。",
        "",
        "原因：物理 k-path metadata 缺失或尚未达到 `physically_verified`。按照 AGENTS 与 Full-2A 约束，Formal Full-2A 需要可追溯 sidecar 且 5-sample WSL `meep_env` MPB physical validation 通过后才能启动。",
        "",
        "允许的下一步是补齐或验证 dataset/k-path metadata；不要启动 Full-2B。",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _write_query_report(run_dir: Path, leaderboard: list[dict[str, Any]], audit: Mapping[str, Any], *, mode: str) -> Path:
    report = Path("docs/results/query_benchmark_gpu_20260618.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    k_path = _as_mapping(audit.get("k_path"))
    lines = [
        "# Query Benchmark GPU",
        "",
        "日期：2026-06-18",
        f"run_dir: `{run_dir}`",
        f"mode: `{mode}`",
        "",
        "本文是 neural surrogate query-form 检查，不是 MPB/Meep physical validation。",
        "",
        "## k coordinate boundary",
        "",
        f"- coordinate_source: `{k_path.get('coordinate_source')}`",
        f"- is_physical: `{k_path.get('is_physical')}`",
        "",
        "## Mean ± std leaderboard",
        "",
        "| protocol | model | seeds | query_mre_bands_2_10_pct | query_mae | runtime ms/query |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in leaderboard:
        lines.append(
            "| {protocol} | {model} | {n_seeds} | {mre:.6f} ± {mre_std:.6f} | {mae:.6f} ± {mae_std:.6f} | {lat:.6f} ± {lat_std:.6f} |".format(
                protocol=row.get("protocol"),
                model=row.get("model"),
                n_seeds=int(row.get("n_seeds", 0)),
                mre=float(row.get("query_mre_bands_2_10_pct_mean", 0.0)),
                mre_std=float(row.get("query_mre_bands_2_10_pct_std", 0.0)),
                mae=float(row.get("query_mae_mean", 0.0)),
                mae_std=float(row.get("query_mae_std", 0.0)),
                lat=float(row.get("query_inference_time_per_query_ms_mean", 0.0)),
                lat_std=float(row.get("query_inference_time_per_query_ms_std", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "结论边界：若 `is_physical=false`，该结果只能作为 index-query preflight，不是正式物理 k-query benchmark。",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_query_preflight(config_path: str | Path) -> Path:
    return _run_query_matrix(config_path, mode="preflight")


def run_query_benchmark(config_path: str | Path) -> Path:
    return _run_query_matrix(config_path, mode="benchmark")


__all__ = ["run_query_audit", "run_query_benchmark", "run_query_preflight"]
