"""CPU-friendly native forward train/eval runner.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
from collections import Counter
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext
from typing import Any, Mapping

import numpy as np

from phc2d.artifacts import save_forward_band_arrays
from phc2d.data import (
    DataRepresentationSpec,
    infer_dataset_schema,
    load_forward_dataset,
    make_input,
    make_target,
    validate_representation_compatibility,
)
from phc2d.forward.config import ForwardRunConfig, load_forward_run_config
from phc2d.models import (
    build_model,
    architecture_fingerprint,
    describe_model_identity,
    get_model_spec,
    state_dict_schema_hash,
)
from phc2d.plot import plot_metric_report
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    write_json,
    write_run_meta,
    setup_logging,
    write_metrics,
    write_resolved_config,
    write_results,
)
from phc2d.training import (
    ResolvedDeviceInfo,
    autocast_context,
    collect_device_metadata,
    create_grad_scaler,
    move_batch_to_device,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    resolve_device,
    synchronize_device,
)


@dataclass(frozen=True)
class _FieldSampleBundle:
    """Small row-based bundle for HDF5 near-field schema smoke."""

    samples: list[dict[str, Any]]
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.samples)


_NEAR_FIELD_OUTPUTS = {"near_field_te", "near_field_tm", "near_field_te_tm"}

def _require_torch() -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError(
            "Native forward train/eval requires optional dependency 'torch'. "
            "Install the train extra or run in WSL Ubuntu conda meep_env."
        ) from exc
    return torch, DataLoader, TensorDataset


def _resolve_runtime_device(config: ForwardRunConfig, torch: Any) -> ResolvedDeviceInfo:
    return resolve_device(config.runtime, torch_module=torch)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _sample_rows(bundle: Any, spec: DataRepresentationSpec) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for idx in range(bundle.n_samples):
        row = {
            "rho": bundle.rho[idx],
            "Wh": bundle.Wh[idx],
            "metadata": bundle.metadata[idx] if idx < len(bundle.metadata) else {},
        }
        inputs.append(np.asarray(make_input(row, spec), dtype=np.float32))
        targets.append(np.asarray(make_target(row, spec), dtype=np.float32))
    x = np.stack(inputs, axis=0).astype(np.float32)
    y = np.stack(targets, axis=0).astype(np.float32)
    if x.ndim == 4 and x.shape[-1] in {1, 2, 3, 4}:
        x = np.transpose(x, (0, 3, 1, 2))
    return x, y


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1", errors="ignore")
    return value


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _sha256(payload: str | bytes) -> str:
    if isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        data = payload
    return hashlib.sha256(data).hexdigest()


def _hash_array(values: Any) -> str:
    arr = np.asarray(values, dtype=np.int64).ravel()
    return _sha256(arr.astype(np.int64).tobytes())


def _dataset_hash(path: Path) -> str:
    return _sha256(path.read_bytes())


def _serialize_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json(value)
    return value


def _split_signature(bundle: Any) -> dict[str, str]:
    return {
        "idx_train_hash": _hash_array(bundle.idx_train),
        "idx_val_hash": _hash_array(bundle.idx_val),
        "idx_test_hash": _hash_array(bundle.idx_test),
    }


def _sample_id_signature(bundle: Any) -> dict[str, str]:
    train_ids = [str(bundle.metadata[int(idx)]["sample_id"]) for idx in bundle.idx_train if int(idx) < len(bundle.metadata)]
    val_ids = [str(bundle.metadata[int(idx)]["sample_id"]) for idx in bundle.idx_val if int(idx) < len(bundle.metadata)]
    test_ids = [str(bundle.metadata[int(idx)]["sample_id"]) for idx in bundle.idx_test if int(idx) < len(bundle.metadata)]
    return {
        "train_sample_ids_hash": _sha256(_canonical_json(train_ids)),
        "val_sample_ids_hash": _sha256(_canonical_json(val_ids)),
        "test_sample_ids_hash": _sha256(_canonical_json(test_ids)),
        "train_sample_id_count": len(train_ids),
        "val_sample_id_count": len(val_ids),
        "test_sample_id_count": len(test_ids),
    }


def _collect_case_meta(case_row: dict[str, Any]) -> dict[str, Any]:
    train_meta = _safe_load_json(Path(case_row["train_run"]) / "run.meta.json")
    eval_meta = _safe_load_json(Path(case_row["eval_run"]) / "run.meta.json")
    return {
        "seed": int(case_row["seed"]),
        "train_run": case_row["train_run"],
        "eval_run": case_row["eval_run"],
        "train_case_log_exists": (Path(case_row["train_run"]) / "run.log").is_file(),
        "eval_case_log_exists": (Path(case_row["eval_run"]) / "run.log").is_file(),
        "train_run_has_pid": "pid" in train_meta,
        "train_run_has_start_timestamp": "start_timestamp" in train_meta,
        "train_run_has_end_timestamp": "end_timestamp" in train_meta,
        "train_run_pid": train_meta.get("pid"),
        "eval_run_has_pid": "pid" in eval_meta,
        "eval_run_has_start_timestamp": "start_timestamp" in eval_meta,
        "eval_run_has_end_timestamp": "end_timestamp" in eval_meta,
        "eval_run_pid": eval_meta.get("pid"),
        "train_run_git_commit": train_meta.get("git_commit"),
        "eval_run_git_commit": eval_meta.get("git_commit"),
        "train_run_has_wall_time_total_s": "wall_time_total_s" in train_meta,
        "eval_run_has_wall_time_total_s": "wall_time_total_s" in eval_meta,
        "train_run_utc": train_meta.get("utc"),
        "eval_run_utc": eval_meta.get("utc"),
        "train_run_start_timestamp": train_meta.get("start_timestamp"),
        "train_run_end_timestamp": train_meta.get("end_timestamp"),
        "eval_run_start_timestamp": eval_meta.get("start_timestamp"),
        "eval_run_end_timestamp": eval_meta.get("end_timestamp"),
    }


def _case_config_hash(config: ForwardRunConfig) -> str:
    payload = config.to_resolved_dict()
    output = payload.get("output")
    if isinstance(output, dict):
        output = dict(output)
        output.pop("run_dir", None)
        output.pop("report", None)
        payload["output"] = output
    return _sha256(_canonical_json(payload))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _load_json_payload(path)
    except Exception:
        return {}


def _state_dict_key_hash(model: Any) -> str:
    return state_dict_schema_hash(model)


def _architecture_fingerprint(model: Any) -> str:
    return architecture_fingerprint(model)


def _parse_jsonish(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = max(1, int(0.8 * n))
    n_val = max(n_train, int(0.9 * n))
    return (
        np.arange(n_train, dtype=np.int64),
        np.arange(n_train, n_val, dtype=np.int64),
        np.arange(n_val, n, dtype=np.int64),
    )


def _load_field_bundle(path: Path, spec: DataRepresentationSpec) -> _FieldSampleBundle:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for near-field HDF5 smoke mode.") from exc

    with h5py.File(path, "r") as handle:
        arrays: dict[str, np.ndarray] = {}
        for name in ("rho", "epsilon", "eps", "efield", "hfield", "freqs", "k_points", "band_index"):
            if name in handle:
                arrays[name] = np.asarray(handle[name], dtype=np.float32)
            if name in {"epsilon", "eps"} and "eps" not in arrays and "epsilon" in arrays:
                arrays["eps"] = arrays["epsilon"]
        attrs: dict[str, Any] = {_decode_scalar(k): _decode_scalar(v) for k, v in handle.attrs.items()}
        metadata = _parse_jsonish(attrs.get("metadata_json", attrs.get("metadata_jsonl", {})))
        if not metadata:
            metadata = {}
        if attrs.get("metadata") is not None and not metadata:
            metadata = _parse_jsonish(attrs.get("metadata"))
        for item in ("source", "boundary_condition", "grid_spacing", "domain_metadata", "polarization"):
            if item in attrs:
                metadata[item] = attrs[item]

    n_samples = 0
    for value in arrays.values():
        if value.ndim > 0:
            n_samples = max(n_samples, int(value.shape[0]))
    if n_samples <= 0:
        raise ValueError(f"Near-field dataset has no samples: {path}")

    idx_train, idx_val, idx_test = _default_split_indices(n_samples)
    samples: list[dict[str, Any]] = []
    for index in range(n_samples):
        row = {
            "metadata": metadata,
            "source": metadata.get("source"),
            "polarization": metadata.get("polarization"),
            "frequency": arrays.get("freqs", [])[index] if "freqs" in arrays else None,
            "k_points": arrays.get("k_points", [])[index] if "k_points" in arrays else None,
            "band_index": arrays.get("band_index", [])[index] if "band_index" in arrays else None,
        }
        for key in ("rho", "epsilon", "eps", "efield", "hfield"):
            if key in arrays:
                row[key] = arrays[key][index]
        samples.append(row)
    if spec.output_target in _NEAR_FIELD_OUTPUTS and not (
        spec.input_repr == "field_conditioned" or spec.input_repr in {"epsilon_grid", "rho_grid", "rho_wedge", "image_binary", "image_epsilon"}
    ):
        raise ValueError("near-field output requires field or field-conditioned input representation in this smoke path")
    return _FieldSampleBundle(
        samples=samples,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


def _split_indices(bundle: Any, split: str) -> np.ndarray:
    key = str(split).lower()
    if key in {"train", "idx_train"}:
        idx = bundle.idx_train
    elif key in {"val", "valid", "validation", "idx_val"}:
        idx = bundle.idx_val
    elif key in {"test", "idx_test"}:
        idx = bundle.idx_test
    else:
        raise ValueError(f"Unknown split: {split}")
    if len(idx) == 0:
        fallback = bundle.idx_val if len(bundle.idx_val) else bundle.idx_train
        return np.asarray(fallback, dtype=np.int64)
    return np.asarray(idx, dtype=np.int64)


def _parameter_count(model: Any) -> int:
    return int(sum(int(param.numel()) for param in model.parameters()))


def _mre_pct(y_true: np.ndarray, y_pred: np.ndarray, *, floor: float) -> float:
    denom = np.maximum(np.abs(y_true), float(floor))
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / denom))


def _is_schema_smoke(config: ForwardRunConfig) -> bool:
    execution = str(config.execution).strip().lower()
    return (not config.training_enabled) or execution in {"schema_only", "schema-smoke"}


def _coerce_payload_numpy(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray, *, floor: float) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    den = float(np.linalg.norm(np.asarray(y_true, dtype=np.float64)))
    den = max(float(den), float(floor))
    return float(np.linalg.norm(diff) / den)


def _flatten_field_values(payload: Any) -> dict[str, np.ndarray]:
    if isinstance(payload, dict):
        return {str(key): _coerce_payload_numpy(value) for key, value in payload.items() if value is not None}
    return {"field": _coerce_payload_numpy(payload)}


def _field_metrics_from_zero_prediction(target_payload: Any, *, floor: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payloads = _flatten_field_values(target_payload)
    metrics: dict[str, Any] = {}
    prediction_map: dict[str, np.ndarray] = {}
    if not payloads:
        return metrics, prediction_map

    field_errors: list[float] = []
    for key, true_value in payloads.items():
        pred = np.zeros_like(true_value, dtype=np.float64)
        prediction_map[key] = pred
        if key.lower().startswith("te"):
            metrics["te_relative_l2_field_error"] = _relative_l2_error(true_value, pred, floor=floor)
        elif key.lower().startswith("tm"):
            metrics["tm_relative_l2_field_error"] = _relative_l2_error(true_value, pred, floor=floor)
        field_errors.append(_relative_l2_error(true_value, pred, floor=floor))
    if field_errors:
        metrics["relative_l2_field_error"] = float(np.mean(field_errors))
    if "te_relative_l2_field_error" not in metrics:
        metrics["te_relative_l2_field_error"] = None
    if "tm_relative_l2_field_error" not in metrics:
        metrics["tm_relative_l2_field_error"] = None
    return metrics, prediction_map


def _shape_map(payload: Any) -> dict[str, list[int]]:
    payloads = _flatten_field_values(payload)
    return {key: list(value.shape) for key, value in payloads.items()}


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    floor: float,
    elapsed_s: float,
) -> dict[str, Any]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    mse = float(np.mean(np.square(err)))
    mae = float(np.mean(np.abs(err)))
    metrics: dict[str, Any] = {
        "mse": mse,
        "mae": mae,
        "rmse": float(math.sqrt(mse)),
        "eval_time_s": float(elapsed_s),
        "inference_time_s": float(elapsed_s),
        "inference_time_per_sample_ms": float(1000.0 * elapsed_s / max(int(y_true.shape[0]), 1)),
        "throughput_samples_per_s": float(int(y_true.shape[0]) / elapsed_s) if elapsed_s > 0 else 0.0,
    }
    if y_true.ndim >= 3:
        metrics["avg_MRE_pct"] = _mre_pct(y_true, y_pred, floor=floor)
        if y_true.shape[-1] >= 2:
            band_slice = slice(1, y_true.shape[-1])
            metrics["avg_MRE_bands_2_10_pct"] = _mre_pct(
                y_true[..., band_slice],
                y_pred[..., band_slice],
                floor=floor,
            )
            metrics["metric_band_range_one_based"] = [2, int(y_true.shape[-1])]
        else:
            metrics["avg_MRE_bands_2_10_pct"] = None
            metrics["metric_band_range_one_based"] = None
    return metrics


def _band_aware_loss(prediction: Any, target: Any, *, torch: Any, floor: float, weights: Mapping[str, Any]) -> Any:
    """Shape-derived band-aware loss for Wh-style outputs."""

    mse_weight = float(weights.get("mse_weight", 1.0))
    relative_weight = float(weights.get("relative_weight", 0.5))
    ordering_weight = float(weights.get("ordering_weight", 0.05))
    smoothness_weight = float(weights.get("smoothness_weight", 0.02))
    loss = mse_weight * torch.mean(torch.square(prediction - target))
    if int(prediction.shape[-1]) > 1:
        pred_main = prediction[..., 1:]
        target_main = target[..., 1:]
        denom = torch.clamp(torch.abs(target_main), min=float(floor))
        loss = loss + relative_weight * torch.mean(torch.abs(pred_main - target_main) / denom)
        ordering = torch.relu(prediction[..., :-1] - prediction[..., 1:])
        loss = loss + ordering_weight * torch.mean(ordering)
    if prediction.ndim >= 3 and int(prediction.shape[-2]) > 1:
        smooth = prediction[..., 1:, :] - prediction[..., :-1, :]
        loss = loss + smoothness_weight * torch.mean(torch.square(smooth))
    return loss


def _make_forward_loss(config: ForwardRunConfig, *, torch: Any) -> Any:
    """Return the configured train loss callable."""

    if config.loss_name in {"mse", "mse_loss"}:
        return torch.nn.MSELoss()
    if config.loss_name == "band_aware":
        return lambda prediction, target: _band_aware_loss(
            prediction,
            target,
            torch=torch,
            floor=config.relative_error_floor,
            weights=config.loss_params,
        )
    raise ValueError(f"Unsupported forward loss: {config.loss_name!r}")


def _make_optimizer(config: ForwardRunConfig, model: Any, *, torch: Any) -> Any:
    """Build the configured optimizer."""

    name = config.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")


def _save_forward_checkpoint(
    path: Path,
    *,
    model: Any,
    config: ForwardRunConfig,
    x: np.ndarray,
    y: np.ndarray,
    device_metadata: Mapping[str, Any],
    epoch: int,
    torch: Any,
) -> None:
    """Save a forward model checkpoint with the unified payload schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": config.model_name,
            "model_params": config.model_params,
            "input_shape": tuple(int(v) for v in x.shape[1:]),
            "output_shape": tuple(int(v) for v in y.shape[1:]),
            "state_dict": model.state_dict(),
            "resolved_config": config.to_resolved_dict(),
            "device_metadata": dict(device_metadata),
            "epoch": int(epoch),
        },
        path,
    )


def _latest_epoch_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    """Return the highest numbered epoch checkpoint in a run directory."""

    best: tuple[int, Path] | None = None
    for path in run_dir.glob("checkpoint_epoch_*.pt"):
        stem = path.stem
        try:
            epoch = int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if best is None or epoch > best[0]:
            best = (epoch, path)
    return best


def _evaluate_model(
    *,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    device: Any,
    device_info: ResolvedDeviceInfo | None = None,
    torch: Any,
    floor: float,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    x_tensor = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
    synchronize_device(device, torch_module=torch)
    start = time.perf_counter()
    with torch.no_grad():
        amp_context = autocast_context(device_info, device, torch_module=torch) if device_info is not None else nullcontext()
        with amp_context:
            pred_tensor = model(x_tensor)
    synchronize_device(device, torch_module=torch)
    pred = pred_tensor.detach().cpu().numpy()
    elapsed = time.perf_counter() - start
    metrics = _compute_metrics(y[indices], pred, floor=floor, elapsed_s=elapsed)
    return metrics, pred


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_run(
    config: ForwardRunConfig,
    action: str,
    *,
    run_dir_override: Path | None = None,
    device_info: ResolvedDeviceInfo | None = None,
) -> tuple[Path, logging.Logger, dict[str, Any]]:
    run_dir = make_run_dir(
        config.experiment_id,
        runs_dir=config.runs_dir,
        run_dir=run_dir_override or config.run_dir,
    )
    run_start = datetime.now().astimezone()
    run_meta = {
        "script": f"phc2d forward {action}",
        "config_path": str(config.source_path),
        "dataset_path": str(config.dataset_path),
        "experiment_id": config.experiment_id,
        "validation_boundary": "local_smoke_no_mpb_meep",
        "pid": os.getpid(),
        "start_timestamp": run_start.isoformat(),
    }
    if device_info is not None:
        run_meta.update(collect_device_metadata(device_info, config.runtime))
    logger = setup_logging(
        log_file=run_dir / RUN_LOG_NAME,
        run_meta=run_meta,
    )
    write_resolved_config(run_dir, config.to_resolved_yaml())
    ensure_figures_dir(run_dir)
    return run_dir, logger, run_meta


def _device_metadata_with_peak(device_info: ResolvedDeviceInfo, torch: Any) -> dict[str, Any]:
    return device_info.metadata(
        peak_gpu_memory_mb=peak_gpu_memory_mb(device_info.torch_device, torch_module=torch)
        if device_info.record_peak_memory
        else None
    )


def _finalize_run_meta(run_dir: Path, run_meta: Mapping[str, Any], *, wall_time_s: float | None = None) -> None:
    payload = dict(run_meta)
    payload["end_timestamp"] = datetime.now().astimezone().isoformat()
    if wall_time_s is not None:
        payload["wall_time_total_s"] = float(wall_time_s)
    write_run_meta(run_dir, payload)


def _identity_payload(
    request_model: str,
    model: Any,
    *,
    model_params: Mapping[str, Any],
    config_hash: str,
    split_hash: str,
    dataset_hash: str,
) -> dict[str, Any]:
    payload = describe_model_identity(
        model,
        request_model,
        model_kwargs=model_params,
    )
    payload.update(
        {
            "config_hash": config_hash,
            "split_hash": split_hash,
            "dataset_hash": dataset_hash,
            "model_source_file": payload["source_file"],
            "state_dict_key_hash": payload["state_dict_schema_hash"],
        }
    )
    return payload


def _merge_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    current = _safe_load_json(path)
    current.update(dict(payload))
    _write_json(path, current)

def _latest_checkpoint_for_config(config: ForwardRunConfig) -> Path | None:
    """Return the newest checkpoint from a prior run with the same experiment id."""

    if config.checkpoint_path is not None:
        return config.checkpoint_path
    candidates = []
    for path in config.runs_dir.glob(f"*/{config.experiment_id}_*/model.pt"):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _load_arrays(config: ForwardRunConfig) -> tuple[Any, np.ndarray, np.ndarray]:
    schema = infer_dataset_schema(config.dataset_path)
    problems = validate_representation_compatibility(schema, config.representation)
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"Config is incompatible with dataset schema: {joined}")
    suffix = config.dataset_path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        if _is_schema_smoke(config):
            raise RuntimeError(
                "Schema-only mode for HDF5 near-field data is handled by _run_forward_schema_smoke."
            )
        raise ValueError("Near-field HDF5 data requires schema-only execution mode in this repository stage.")
    bundle = load_forward_dataset(config.dataset_path)
    x, y = _sample_rows(bundle, config.representation)
    return bundle, x, y


def _sample_field_rows(bundle: _FieldSampleBundle, spec: DataRepresentationSpec) -> tuple[Any, Any]:
    samples: list[Any] = []
    targets: list[Any] = []
    for idx in range(bundle.n_samples):
        row = dict(bundle.samples[idx])
        input_payload = make_input(row, spec)
        target_payload = make_target(row, spec)
        samples.append(input_payload)
        targets.append(target_payload)
    return samples, targets


def _save_metric_figure(run_dir: Path, metrics: Mapping[str, Any]) -> None:
    try:
        plot_metric_report(metrics, run_dir / "figures" / "metrics.png", title="Forward smoke metrics")
    except Exception as exc:  # pragma: no cover - figure is auxiliary, artifacts still valid
        logging.getLogger(__name__).warning("Could not write metric figure: %s", exc)


def _write_field_report(
    run_dir: Path,
    *,
    config: ForwardRunConfig,
    schema: Mapping[str, Any],
    sample_shape: Mapping[str, Any],
    metrics: Mapping[str, Any],
    status: str,
    report_path: Path | None = None,
) -> None:
    lines = [
        "# Near-field feasibility smoke report",
        "",
        f"- experiment_id: {config.experiment_id}",
        f"- dataset_path: {config.dataset_path}",
        f"- run_dir: {run_dir}",
        f"- status: {status}",
        f"- representation: {config.representation.input_repr} -> {config.representation.output_target}",
        "",
        "## dataset schema",
        f"- kind: {schema.get('kind')}",
        f"- n_samples: {schema.get('n_samples')}",
        f"- rho_shape: {schema.get('rho_shape')}",
        f"- grid_shape: {schema.get('grid_shape')}",
        f"- k_points: {schema.get('k_points')}",
        f"- n_bands: {schema.get('n_bands')}",
        "",
        "## sample payload shape",
    ]
    for key, value in sample_shape.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## metric smoke",
            *(f"- {key}: {value}" for key, value in metrics.items()),
        ]
    )
    report_payload = "\n".join(lines)
    # Keep legacy artifact name for compatibility.
    (run_dir / "near_field_feasibility.md").write_text(report_payload, encoding="utf-8")

    if report_path is None:
        return
    if report_path.name.strip():
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_payload, encoding="utf-8")


def _run_forward_schema_smoke(
    config: ForwardRunConfig,
    *,
    run_dir_override: Path | None = None,
) -> Path:
    wall_start = time.perf_counter()
    run_dir, logger, run_meta = _prepare_run(config, action="schema_smoke", run_dir_override=run_dir_override)
    logger.info("Starting near-field schema-only smoke; this is not MPB/Meep validation.")

    schema = infer_dataset_schema(config.dataset_path)
    problems = validate_representation_compatibility(schema, config.representation)
    if problems:
        joined = "; ".join(problems)
        raise ValueError(f"Config is incompatible with dataset schema: {joined}")

    bundle = _load_field_bundle(config.dataset_path, config.representation)
    if bundle.n_samples <= 0:
        raise ValueError(f"Near-field dataset has no valid samples: {config.dataset_path}")

    sample_row = bundle.samples[0]
    input_payload = make_input(sample_row, config.representation)
    target_payload = make_target(sample_row, config.representation)
    if target_payload is None:
        raise ValueError(
            f"Output target '{config.representation.output_target}' is unavailable for first sample in {config.dataset_path}"
        )

    start = time.perf_counter()
    field_metrics, _ = _field_metrics_from_zero_prediction(
        target_payload,
        floor=config.relative_error_floor,
    )
    inference_time_s = time.perf_counter() - start

    payload_shapes = {
        "input": list(np.asarray(input_payload).shape)
        if not isinstance(input_payload, dict)
        else {
            str(key): (
                list(np.asarray(value).shape)
                if isinstance(value, (list, tuple, np.ndarray))
                else [type(value).__name__]
            )
            for key, value in input_payload.items()
        },
        "target": _shape_map(target_payload),
    }

    metrics = dict(field_metrics)
    metrics.update(
        {
            "field_inference_time_s": float(inference_time_s),
            "field_inference_time_per_sample_ms": float(1000.0 * inference_time_s),
            "setup_time_s": 0.0,
            "wall_time_total_s": float(time.perf_counter() - wall_start),
            "model_not_trained": True,
            "n_samples": int(bundle.n_samples),
            "n_train": int(len(_split_indices(bundle, "train"))),
            "n_val": int(len(_split_indices(bundle, "val"))),
            "n_test": int(len(_split_indices(bundle, "test"))),
        }
    )
    _write_history(
        run_dir / "history.csv",
        [
            {
                "kind": "schema_smoke",
                "sample_index": 0,
                "wall_time_total_s": metrics["wall_time_total_s"],
                "field_inference_time_s": metrics["field_inference_time_s"],
            }
        ],
    )
    write_metrics(run_dir, metrics)
    write_results(
        run_dir,
        {
            "status": "ok",
            "stage": "forward_train_schema_only",
            "run_dir": str(run_dir),
            "smoke_only_no_mpb_meep": True,
            "dataset_path": str(config.dataset_path),
            "model_name": config.model_name,
            "schema_only": True,
            "validation_boundary": "local_smoke_no_mpb_meep",
            "metrics": {key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        },
    )
    _write_field_report(
        run_dir,
        config=config,
        schema=schema,
        sample_shape=payload_shapes,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None},
        status="ok",
        report_path=Path(config.report_path) if config.report_path else None,
    )
    _save_metric_figure(run_dir, metrics)
    logger.info("Finished near-field schema smoke: %s", run_dir)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    return run_dir


def run_forward_train(config_path: str | Path) -> Path:
    """Train a registered forward model on a tiny/smoke NPZ dataset."""
    from phc2d.forward.field_eigen_v2 import is_field_eigen_v2_config, run_field_eigen_v2_train
    if is_field_eigen_v2_config(config_path):
        return run_field_eigen_v2_train(config_path)
    from phc2d.forward.field_eigen import is_field_eigen_config, run_field_eigen_train
    if is_field_eigen_config(config_path):
        return run_field_eigen_train(config_path)
    return _run_forward_train(load_forward_run_config(config_path))


def run_forward_eval(config_path: str | Path) -> Path:
    """Evaluate a frozen forward model or checkpoint on a configured split."""
    from phc2d.forward.field_eigen_v2 import is_field_eigen_v2_config, run_field_eigen_v2_eval
    if is_field_eigen_v2_config(config_path):
        return run_field_eigen_v2_eval(config_path)
    from phc2d.forward.field_eigen import is_field_eigen_config, run_field_eigen_eval
    if is_field_eigen_config(config_path):
        return run_field_eigen_eval(config_path)
    return _run_forward_eval(load_forward_run_config(config_path))


def run_forward_benchmark(config_path: str | Path) -> Path:
    """Run a small representation-and-model benchmark matrix."""
    from phc2d.forward.field_eigen_b4 import is_field_eigen_b4_dense_config, run_field_eigen_b4_dense_gate
    if is_field_eigen_b4_dense_config(config_path):
        return run_field_eigen_b4_dense_gate(config_path)
    from phc2d.forward.field_eigen_v2 import is_field_eigen_v2_config, run_field_eigen_v2_benchmark
    if is_field_eigen_v2_config(config_path):
        return run_field_eigen_v2_benchmark(config_path)
    from phc2d.forward.field_eigen import is_field_eigen_config, run_field_eigen_benchmark
    if is_field_eigen_config(config_path):
        return run_field_eigen_benchmark(config_path)
    wall_start = time.perf_counter()
    config = load_forward_run_config(config_path)
    raw = _as_mapping(config.raw, name="config")
    fair_selection = _as_mapping(raw.get("fair_selection"), name="fair_selection")
    if bool(fair_selection.get("enabled", False)):
        return _run_forward_fair_selection(config, raw)
    experiment = _as_mapping(raw.get("experiment"), name="experiment")
    stage = str(experiment.get("stage", "")).lower()
    if config.representation.output_target == "band_query" or "query" in stage:
        from phc2d.query import run_query_benchmark

        return run_query_benchmark(config_path)
    torch, _, _ = _require_torch()
    benchmark_cases = _parse_benchmark_cases(config, raw)
    if not benchmark_cases:
        raise ValueError("Benchmark config has no valid cases; add entries under model.cases/benchmark/matrix.")
    alias_cases = []
    for case in benchmark_cases:
        spec = get_model_spec(str(case["model"]))
        if spec.alias_of is not None:
            alias_cases.append(f"{spec.name}->alias_of:{spec.alias_of}")
    if alias_cases:
        joined = ", ".join(alias_cases)
        raise ValueError(
            "Benchmark matrices cannot include alias models as independent leaderboard cases. "
            f"Use the canonical model instead: {joined}"
        )

    prior_benchmark_runs = sorted(
        [p for p in config.runs_dir.glob(f"*/{config.experiment_id}_*") if p.is_dir()],
        key=lambda path: str(path),
    )
    device_info = _resolve_runtime_device(config, torch)
    eval_device = device_info.torch_device
    reset_peak_gpu_memory(eval_device, torch_module=torch)
    run_dir, logger, run_meta = _prepare_run(
        config,
        action="forward benchmark",
        run_dir_override=config.run_dir,
        device_info=device_info,
    )
    logger.info("Starting local forward benchmark; this is not MPB/Meep validation.")
    write_resolved_config(run_dir, config.to_resolved_yaml())

    schema = infer_dataset_schema(config.dataset_path)
    seeds = tuple(int(seed) for seed in getattr(config, "seeds", (config.seed,)))
    if not seeds:
        seeds = (int(config.seed),)
    learning_rates = tuple(float(lr) for lr in getattr(config, "learning_rates", ()))
    if not learning_rates:
        learning_rates = (float(config.learning_rate),)
    case_rows: list[dict[str, Any]] = []
    per_band_rows: list[dict[str, Any]] = []
    model_identity_rows: list[dict[str, Any]] = []
    bundle = load_forward_dataset(config.dataset_path)
    failed_cases = 0
    split_signature = _split_signature(bundle)
    split_hash = _sha256(_canonical_json(split_signature))
    sample_signature = _sample_id_signature(bundle)
    dataset_hash = _dataset_hash(config.dataset_path)
    expected_seed_set = set(seeds)
    expected_case_rows = len(benchmark_cases) * len(seeds) * len(learning_rates)
    sample_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    reference_transform_signature = {
        "normalization": dict(config.representation.normalization),
        "input_repr": str(config.representation.input_repr),
        "output_target": str(config.representation.output_target),
        "relative_error_floor": float(config.relative_error_floor),
        "split": str(config.eval_split),
    }
    metric_field_order = (
        "avg_MRE_pct",
        "avg_MRE_bands_2_10_pct",
        "inference_time_per_sample_ms",
        "throughput_samples_per_s",
        "setup_time_s",
        "parameter_count",
        "wall_time_total_s",
        "train_time_s",
        "eval_time_s",
        "resolved_device",
        "device_name",
        "precision",
        "peak_gpu_memory_mb",
        "batch_size",
        "effective_batch_size",
        "num_workers",
        "pin_memory",
        "non_blocking_transfer",
        "requested_device",
        "accelerator_required",
    )
    case_output_rows: list[dict[str, Any]] = []

    for case_index, case in enumerate(benchmark_cases, start=1):
        case_input = str(case["input_repr"])
        case_model = str(case["model"])
        case_params = dict(case.get("model_params", {}))
        case_params = _filter_model_params(case_model, case_params)
        case_output = str(case.get("output_target", config.representation.output_target))
        case_spec = replace(config.representation, input_repr=case_input, output_target=case_output)
        case_problems = validate_representation_compatibility(schema, case_spec)
        if case_problems:
            joined = "; ".join(case_problems)
            raise ValueError(f"Benchmark case '{case_input}_{case_model}_{case_index}' is incompatible: {joined}")

        base_case_id = f"{case_input}_{case_model}_{case_index}"
        if case_output != str(config.representation.output_target):
            base_case_id = f"{base_case_id}_{case_output}"
        sample_cache_key = (case_input, case_output)
        if sample_cache_key not in sample_cache:
            sample_cache[sample_cache_key] = _sample_rows(bundle, case_spec)
        x, y = sample_cache[sample_cache_key]
        model_kwargs_base = dict(
            case_params,
            input_shape=tuple(int(v) for v in x.shape[1:]),
            output_shape=tuple(int(v) for v in y.shape[1:]),
        )

        for learning_rate in learning_rates:
            lr_label = f"lr{float(learning_rate):.0e}".replace("+", "").replace("-", "m")
            lr_case_id = base_case_id if len(learning_rates) == 1 else f"{base_case_id}_{lr_label}"
            for seed in seeds:
                seed_case_id = lr_case_id if len(seeds) == 1 else f"{lr_case_id}_seed{seed}"
                case_dir = run_dir / "cases" / seed_case_id
                case_dataset = replace(
                    config,
                    experiment_id=f"{config.experiment_id}_{seed_case_id}",
                    representation=case_spec,
                    model_name=case_model,
                    model_params=case_params,
                    seed=int(seed),
                    learning_rate=float(learning_rate),
                )
                seed_row: dict[str, Any] = {
                    "base_case_id": base_case_id,
                    "case_id": seed_case_id,
                    "seed": int(seed),
                    "learning_rate": float(learning_rate),
                    "input_repr": case_input,
                    "output_target": case_output,
                    "model": case_model,
                    "status": "failed",
                    "dataset_hash": dataset_hash,
                    "idx_train_hash": split_signature["idx_train_hash"],
                    "idx_val_hash": split_signature["idx_val_hash"],
                    "idx_test_hash": split_signature["idx_test_hash"],
                    "sample_id_signature": sample_signature,
                    "split_hash": split_hash,
                    "split_signature": split_signature,
                    "dataset_path": str(config.dataset_path),
                    "train_run_dir": str(case_dir / "train"),
                    "eval_run_dir": str(case_dir / "eval"),
                }
                try:
                    _set_seed(seed, torch)
                    train_run = case_dir / "train"
                    eval_run = case_dir / "eval"
                    checkpoint = train_run / "model.pt"
                    resumed_case = checkpoint.is_file() and (eval_run / "metrics.json").is_file()
                    if not resumed_case:
                        train_run = _run_forward_train(
                            case_dataset,
                            run_dir_override=case_dir / "train",
                            torch_module=torch,
                        )
                        checkpoint = train_run / "model.pt"
                        eval_config = replace(case_dataset, checkpoint_path=checkpoint)
                        eval_run = _run_forward_eval(
                            eval_config,
                            run_dir_override=case_dir / "eval",
                            torch_module=torch,
                        )
                    else:
                        eval_config = replace(case_dataset, checkpoint_path=checkpoint)

                    eval_metrics = _load_json_payload(eval_run / "metrics.json")
                    train_metrics = _load_json_payload(train_run / "metrics.json")
                    eval_idx = _split_indices(bundle, case_dataset.eval_split)
                    eval_model, _ = _load_eval_model(
                        eval_config,
                        tuple(int(v) for v in x.shape[1:]),
                        tuple(int(v) for v in y.shape[1:]),
                        torch,
                        eval_device,
                        checkpoint_path=checkpoint,
                    )
                    _, prediction = _evaluate_model(
                        model=eval_model,
                        x=x,
                        y=y,
                        indices=eval_idx,
                        device=eval_device,
                        device_info=device_info,
                        torch=torch,
                        floor=config.relative_error_floor,
                    )
                    eval_band_metrics = _compute_per_band_metrics(
                        y[eval_idx],
                        prediction,
                        floor=config.relative_error_floor,
                    )
                    per_band_rows.extend(
                        {
                            "case_id": seed_case_id,
                            "base_case_id": base_case_id,
                            "seed": int(seed),
                            "learning_rate": float(learning_rate),
                            "input_repr": case_input,
                            "output_target": case_output,
                            "model": case_model,
                            "band": int(item["band"]),
                            **item,
                        }
                        for item in eval_band_metrics
                    )
                    case_config_hash = _case_config_hash(eval_config)
                    model_identity = _identity_payload(
                        request_model=case_model,
                        model=eval_model,
                        model_params=model_kwargs_base,
                        config_hash=case_config_hash,
                        split_hash=split_hash,
                        dataset_hash=dataset_hash,
                    )
                    seed_row.update(
                        {
                            "status": "ok",
                            "resumed_from_case_artifacts": bool(resumed_case),
                            "train_run": str(train_run),
                            "eval_run": str(eval_run),
                            "n_train": int(train_metrics.get("n_train", 0)),
                            "n_eval": int(eval_metrics.get("n_eval", 0)),
                            "wall_time_total_s": float(eval_metrics.get("wall_time_total_s", 0.0)),
                            "train_time_s": float(train_metrics.get("train_time_s", 0.0)),
                            "eval_time_s": float(eval_metrics.get("eval_time_s", 0.0)),
                            "mse": float(eval_metrics.get("mse", 0.0)),
                            "mae": float(eval_metrics.get("mae", 0.0)),
                            "rmse": float(eval_metrics.get("rmse", 0.0)),
                            "config_hash": case_config_hash,
                            "split_hash": split_hash,
                            "config_metric_domain": reference_transform_signature,
                            "requested_model_name": model_identity["requested_model_name"],
                            "canonical_model_name": model_identity["canonical_model_name"],
                            "model_implementation_module": model_identity["implementation_module"],
                            "model_implementation_class": model_identity["implementation_class"],
                            "model_factory": model_identity["registry_factory"],
                            "model_source_file": model_identity["model_source_file"],
                            "model_kwargs": model_identity["model_kwargs"],
                            "state_dict_key_names": model_identity["state_dict_key_names"],
                            "state_dict_schema_hash": model_identity["state_dict_schema_hash"],
                            "state_dict_key_hash": model_identity["state_dict_schema_hash"],
                            "factory_function_identity": model_identity["factory_function_identity"],
                            "architecture_fingerprint": model_identity["architecture_fingerprint"],
                            "parameter_count": model_identity["parameter_count"],
                            "trainable_parameter_count": model_identity["trainable_parameter_count"],
                            "checkpoint_path": str(checkpoint),
                            "requested_device": eval_metrics.get("requested_device"),
                            "resolved_device": eval_metrics.get("resolved_device"),
                            "accelerator_required": eval_metrics.get("accelerator_required"),
                            "device_name": eval_metrics.get("device_name"),
                            "device_index": eval_metrics.get("device_index"),
                            "torch_version": eval_metrics.get("torch_version"),
                            "torch_cuda_version": eval_metrics.get("torch_cuda_version"),
                            "cudnn_version": eval_metrics.get("cudnn_version"),
                            "precision": eval_metrics.get("precision"),
                            "amp_enabled": eval_metrics.get("amp_enabled"),
                            "amp_dtype": eval_metrics.get("amp_dtype"),
                            "tf32_enabled": eval_metrics.get("tf32_enabled"),
                            "peak_gpu_memory_mb": eval_metrics.get("peak_gpu_memory_mb"),
                            "batch_size": eval_metrics.get("batch_size"),
                            "effective_batch_size": eval_metrics.get("effective_batch_size"),
                            "num_workers": eval_metrics.get("num_workers"),
                            "pin_memory": eval_metrics.get("pin_memory"),
                            "non_blocking_transfer": eval_metrics.get("non_blocking_transfer"),
                        }
                    )
                    for extra in metric_field_order:
                        extra_value = eval_metrics.get(extra)
                        if isinstance(extra_value, (int, float)):
                            seed_row[extra] = float(extra_value)
                        elif extra_value is not None:
                            seed_row[extra] = extra_value
                        elif extra not in seed_row:
                            seed_row[extra] = None
                    identity_update = {"model_identity": model_identity}
                    _merge_json_file(train_run / "run.meta.json", identity_update)
                    _merge_json_file(train_run / "results.json", identity_update)
                    _merge_json_file(eval_run / "run.meta.json", identity_update)
                    _merge_json_file(eval_run / "results.json", identity_update)
                    model_identity_rows.append(
                        {
                            "case_id": seed_case_id,
                            "base_case_id": base_case_id,
                            "seed": int(seed),
                            "learning_rate": float(learning_rate),
                            "input_repr": case_input,
                            "output_target": case_output,
                            **model_identity,
                        }
                    )
                except Exception as exc:
                    failed_cases += 1
                    seed_row["error"] = str(exc)
                    logging.getLogger(__name__).exception(
                        "Benchmark case failed (case=%s, seed=%s)", seed_case_id, seed
                    )
                case_rows.append(seed_row)
                if seed_row["status"] == "ok":
                    case_output_rows.append(seed_row)

    leaderboard_path = run_dir / "leaderboard.csv"
    _write_rows_csv(leaderboard_path, case_rows)
    _write_history(run_dir / "history.csv", case_rows)
    _write_rows_csv(run_dir / "per_band_metrics.csv", per_band_rows)
    _write_json(
        run_dir / "model_identity_summary.json",
        {
            "config_path": str(config.source_path),
            "dataset_path": str(config.dataset_path),
            "identity_rows": model_identity_rows,
            "unique_architecture_fingerprints": sorted(
                {
                    str(row["architecture_fingerprint"])
                    for row in model_identity_rows
                    if row.get("architecture_fingerprint")
                }
            ),
            "alias_cases": [],
        },
    )

    case_run_meta = [_collect_case_meta(row) for row in case_output_rows]
    case_ids = [row["case_id"] for row in case_rows]
    base_case_id_counts = Counter(row["base_case_id"] for row in case_rows)
    seed_coverage = {base_case_id: sorted({int(row["seed"]) for row in case_rows if row.get("base_case_id") == base_case_id}) for base_case_id in base_case_id_counts}
    train_case_dirs = [row["train_run"] for row in case_rows if row.get("train_run")]
    eval_case_dirs = [row["eval_run"] for row in case_rows if row.get("eval_run")]
    checkpoint_paths = [row.get("checkpoint_path") for row in case_rows if row.get("checkpoint_path")]
    baseline_transform = reference_transform_signature
    transform_checks = {
        "normalization": all(
            row.get("config_metric_domain", {}).get("normalization") == baseline_transform["normalization"]
            for row in case_rows
        ),
        "relative_error_floor": all(
            row.get("config_metric_domain", {}).get("relative_error_floor") == baseline_transform["relative_error_floor"]
            for row in case_rows
        ),
        "input_repr": all(
            row.get("config_metric_domain", {}).get("input_repr") == baseline_transform["input_repr"]
            for row in case_rows
        ),
        "output_target": all(
            row.get("config_metric_domain", {}).get("output_target") == baseline_transform["output_target"]
            for row in case_rows
        ),
        "split_signature": all(
            row.get("split_signature") == split_signature for row in case_rows
        ),
        "sample_signature": all(
            row.get("sample_id_signature") == sample_signature for row in case_rows
        ),
    }

    latest_runs = sorted([p for p in config.runs_dir.glob(f"*/{config.experiment_id}_*") if p.is_dir()], key=lambda path: str(path))
    run_meta_payload = _safe_load_json(run_dir / "run.meta.json")
    run_case_pid_map = [row.get("train_run_pid") for row in case_run_meta] + [row.get("eval_run_pid") for row in case_run_meta]
    run_case_pid_map = [pid for pid in run_case_pid_map if pid is not None]
    process_audit = {
        "run_dir": str(run_dir),
        "run_meta_pid_present": "pid" in run_meta_payload,
        "run_meta_start_timestamp_present": "start_timestamp" in run_meta_payload,
        "run_meta_end_timestamp_present": "end_timestamp" in run_meta_payload,
        "run_meta_pid": run_meta_payload.get("pid"),
        "run_meta_start_timestamp": run_meta_payload.get("start_timestamp"),
        "run_meta_end_timestamp": run_meta_payload.get("end_timestamp"),
        "case_runs_with_pid": len(run_case_pid_map),
        "case_run_unique_pids": len(set(run_case_pid_map)),
        "case_run_pid_values": sorted({int(pid) for pid in run_case_pid_map}) if run_case_pid_map else [],
        "has_train_eval_pid_mismatch": any(
            (row.get("train_run_pid") is not None and row.get("eval_run_pid") is not None and row["train_run_pid"] != row["eval_run_pid"])
            for row in case_run_meta
        ),
        "multiple_benchmark_dirs_detected": len(latest_runs) > 1,
        "all_benchmark_dirs": [str(item) for item in latest_runs],
        "prior_benchmark_dirs": [str(item) for item in prior_benchmark_runs if item != run_dir],
    }

    integrity_payload = {
        "run_dir": str(run_dir),
        "run_name": f"{config.experiment_id}_{run_meta.get('start_timestamp', '')}",
        "config_path": str(config.source_path),
        "dataset_path": str(config.dataset_path),
        "dataset_hash": dataset_hash,
        "split_signature": split_signature,
        "sample_signature": sample_signature,
        "benchmark_wall_time_total_s": float(time.perf_counter() - wall_start),
        "expected_case_rows": expected_case_rows,
        "actual_case_rows": len(case_rows),
        "learning_rates": [float(item) for item in learning_rates],
        "n_failures": int(failed_cases),
        "leaderboard_rows": len(case_rows),
        "base_case_count": len(base_case_id_counts),
        "seed_coverage": {key: value for key, value in seed_coverage.items()},
        "cases_with_incomplete_seeds": {
            key: sorted(value)
            for key, value in seed_coverage.items()
            if set(value) != expected_seed_set
        },
        "duplicate_case_ids": [entry for entry, count in Counter(case_ids).items() if count > 1],
        "train_case_dir_collision": len(set(train_case_dirs)) < len(train_case_dirs),
        "eval_case_dir_collision": len(set(eval_case_dirs)) < len(eval_case_dirs),
        "checkpoint_collision": len(set(checkpoint_paths)) < len(checkpoint_paths),
        "case_run_meta": case_run_meta,
        "case_run_meta_presence_checks": {
            "train_eval_metas_present": len(case_run_meta),
            "all_cases_have_case_logs": all(
                row.get("train_case_log_exists") and row.get("eval_case_log_exists") for row in case_run_meta
            ),
            "all_case_metas_have_pid": all(
                row.get("train_run_has_pid") and row.get("eval_run_has_pid") for row in case_run_meta
            ),
            "all_case_metas_have_start_timestamp": all(
                row.get("train_run_has_start_timestamp") and row.get("eval_run_has_start_timestamp")
                for row in case_run_meta
            ),
            "all_case_metas_have_end_timestamp": all(
                row.get("train_run_has_end_timestamp") and row.get("eval_run_has_end_timestamp")
                for row in case_run_meta
            ),
        },
        "transform_consistency": transform_checks,
        "integrity_checks": {
            "benchmark_case_row_count_matches_expected": len(case_rows) == expected_case_rows,
            "base_case_count_matches_config": len(base_case_id_counts) == len(benchmark_cases),
            "all_base_cases_have_expected_seeds": all(
                set(values) == expected_seed_set for values in seed_coverage.values()
            ),
            "no_output_dir_collision": len(set(train_case_dirs)) == len(train_case_dirs)
            and len(set(eval_case_dirs)) == len(eval_case_dirs),
            "no_checkpoint_collision": len(set(checkpoint_paths)) == len(checkpoint_paths),
            "all_case_metas_have_pid": all(
                row.get("train_run_has_pid") and row.get("eval_run_has_pid")
                for row in case_run_meta
            )
            if case_run_meta
            else False,
            "all_case_metas_have_timestamps": all(
                row.get("train_run_has_start_timestamp")
                and row.get("train_run_has_end_timestamp")
                and row.get("eval_run_has_start_timestamp")
                and row.get("eval_run_has_end_timestamp")
                for row in case_run_meta
            )
            if case_run_meta
            else False,
            "all_cases_consistent_split": all(row.get("split_signature") == split_signature for row in case_rows),
            "all_cases_consistent_samples": all(
                row.get("sample_id_signature") == sample_signature for row in case_rows
            ),
            "all_cases_consistent_metric_domain": all(
                row.get("config_metric_domain") == reference_transform_signature for row in case_rows
            ),
        },
        "process_audit": process_audit,
    }
    _write_json(run_dir / "integrity_audit.json", integrity_payload)
    _write_json(run_dir / "integrity_summary.json", integrity_payload)

    benchmark_metrics = {
        "experiment_id": config.experiment_id,
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "cases": len(case_rows),
        "results_dir": str(run_dir),
        "dataset_path": str(config.dataset_path),
        "n_failures": int(failed_cases),
        **_device_metadata_with_peak(device_info, torch),
    }
    write_metrics(run_dir, benchmark_metrics)
    write_results(
        run_dir,
        {
            "status": "ok",
            "stage": "forward_benchmark",
            "run_dir": str(run_dir),
            "metric_rows": len(case_rows),
            "case_rows": [row["case_id"] for row in case_rows],
            "smoke_only_no_mpb_meep": True,
        },
    )
    report_path = _write_benchmark_report(config, case_rows, per_band_rows, completed_at=datetime.now())
    logger.info("Benchmark finished: %s; report=%s", run_dir, report_path)
    _save_metric_figure(run_dir, benchmark_metrics)
    run_meta.update(benchmark_metrics)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    finalized_meta = _safe_load_json(run_dir / "run.meta.json")
    process_audit["run_meta_end_timestamp_present"] = "end_timestamp" in finalized_meta
    process_audit["run_meta_end_timestamp"] = finalized_meta.get("end_timestamp")
    integrity_payload["process_audit"] = process_audit
    _write_json(run_dir / "integrity_audit.json", integrity_payload)
    _write_json(run_dir / "integrity_summary.json", integrity_payload)
    return run_dir


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return the strict three-seed summary used by A1 reports."""

    if not values:
        return (float("nan"), float("nan"))
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def _run_forward_fair_selection(config: ForwardRunConfig, raw: Mapping[str, Any]) -> Path:
    """Run a validation-selected forward matrix without test-driven LR choice."""

    wall_start = time.perf_counter()
    torch, _, _ = _require_torch()
    device_info = _resolve_runtime_device(config, torch)
    run_dir, logger, run_meta = _prepare_run(config, action="forward fair benchmark", run_dir_override=config.run_dir, device_info=device_info)
    write_resolved_config(run_dir, config.to_resolved_yaml())
    cases = _parse_benchmark_cases(config, raw)
    if not cases:
        raise ValueError("Fair forward benchmark requires an explicit benchmark matrix.")
    seeds = tuple(int(item) for item in config.seeds)
    learning_rates = tuple(float(item) for item in config.learning_rates) or (float(config.learning_rate),)
    if len(seeds) != 3:
        raise ValueError("Fair forward benchmark requires exactly three configured seeds.")
    bundle = load_forward_dataset(config.dataset_path)
    dataset_hash = _dataset_hash(config.dataset_path)
    split_signature = _split_signature(bundle)
    fair_cfg = _as_mapping(raw.get("fair_selection"), name="fair_selection")
    expected_hash = fair_cfg.get("expected_dataset_sha256")
    if expected_hash and str(expected_hash) != dataset_hash:
        raise ValueError("Fair forward benchmark dataset hash does not match the frozen protocol.")
    expected_sizes = _as_mapping(fair_cfg.get("expected_split_sizes"), name="fair_selection.expected_split_sizes")
    actual_sizes = {"train": len(bundle.idx_train), "val": len(bundle.idx_val), "test": len(bundle.idx_test)}
    if expected_sizes and any(int(expected_sizes[name]) != actual_sizes[name] for name in actual_sizes):
        raise ValueError("Fair forward benchmark split sizes do not match the frozen protocol.")
    validation_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    per_band_rows: list[dict[str, Any]] = []
    model_identity_rows: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        model_name = str(case["model"])
        input_repr = str(case["input_repr"])
        output_target = str(case.get("output_target", config.representation.output_target))
        model_params = _filter_model_params(model_name, dict(case.get("model_params", {})))
        representation = replace(config.representation, input_repr=input_repr, output_target=output_target)
        base_case_id = f"{input_repr}_{model_name}_{case_index}"
        lr_rows: dict[float, list[dict[str, Any]]] = {}
        for learning_rate in learning_rates:
            for seed in seeds:
                case_id = f"{base_case_id}_lr{learning_rate:.0e}_seed{seed}".replace("-", "m")
                train_config = replace(
                    config,
                    experiment_id=f"{config.experiment_id}_{case_id}", representation=representation,
                    model_name=model_name, model_params=model_params, seed=seed, learning_rate=learning_rate,
                    eval_split="val", run_dir=run_dir / "cases" / case_id / "train",
                )
                train_dir = _run_forward_train(train_config, run_dir_override=train_config.run_dir)
                metrics = _load_json_payload(train_dir / "metrics.json")
                row = {
                    "phase": "validation", "base_case_id": base_case_id, "case_id": case_id,
                    "model": model_name, "input_repr": input_repr, "output_target": output_target,
                    "seed": seed, "learning_rate": learning_rate, "checkpoint_path": str(train_dir / "model.pt"),
                    "train_run_dir": str(train_dir), "dataset_hash": dataset_hash, "split_signature": split_signature,
                    **metrics,
                }
                validation_rows.append(row)
                lr_rows.setdefault(learning_rate, []).append(row)
        candidates = []
        for learning_rate, rows in lr_rows.items():
            valid = [row for row in rows if row.get("avg_MRE_bands_2_10_pct") is not None]
            if len(valid) != len(seeds):
                continue
            mean, std = _mean_std([float(row["avg_MRE_bands_2_10_pct"]) for row in valid])
            candidates.append((mean, learning_rate, std, valid))
        if not candidates:
            raise RuntimeError(f"No complete three-seed validation candidate for {base_case_id}.")
        mean, selected_lr, std, selected_rows = min(candidates, key=lambda item: item[0])
        selection = {
            "base_case_id": base_case_id, "model": model_name, "input_repr": input_repr,
            "output_target": output_target, "selected_learning_rate": selected_lr,
            "selection_metric": "avg_MRE_bands_2_10_pct", "validation_mean": mean,
            "validation_std": std, "seed_count": len(selected_rows),
            "selection_split": "val", "final_evaluation_split": "test",
            "checkpoint_paths": [row["checkpoint_path"] for row in selected_rows],
        }
        selections.append(selection)
        for selected in selected_rows:
            test_config = replace(
                config, experiment_id=f"{config.experiment_id}_{selected['case_id']}_test",
                representation=representation, model_name=model_name, model_params=model_params,
                seed=int(selected["seed"]), learning_rate=selected_lr, eval_split="test",
                checkpoint_path=Path(str(selected["checkpoint_path"])), run_dir=run_dir / "cases" / str(selected["case_id"]) / "test",
            )
            test_dir = _run_forward_eval(test_config, run_dir_override=test_config.run_dir)
            metrics = _load_json_payload(test_dir / "metrics.json")
            _, x, y = _load_arrays(test_config)
            identity_model = build_model(model_name, input_shape=tuple(int(value) for value in x.shape[1:]), output_shape=tuple(int(value) for value in y.shape[1:]), **model_params)
            model_identity = _identity_payload(model_name, identity_model, model_params=model_params, config_hash=_case_config_hash(test_config), split_hash=_sha256(_canonical_json(split_signature)), dataset_hash=dataset_hash)
            test_row = {**selected, "phase": "test", "test_run_dir": str(test_dir), **metrics, **model_identity}
            test_rows.append(test_row)
            model_identity_rows.append({"base_case_id": base_case_id, "seed": int(selected["seed"]), "learning_rate": selected_lr, **model_identity})
            arrays = _load_json_payload(test_dir / "results.json").get("band_array_artifacts", {})
            target_path = arrays.get("target_bands")
            pred_path = arrays.get("surrogate_pred_bands")
            if target_path and pred_path:
                target_file = Path(str(target_path.get("path"))) if isinstance(target_path, Mapping) else Path(str(target_path))
                pred_file = Path(str(pred_path.get("path"))) if isinstance(pred_path, Mapping) else Path(str(pred_path))
                if not target_file.is_absolute():
                    target_file = test_dir / target_file
                if not pred_file.is_absolute():
                    pred_file = test_dir / pred_file
                per_band_rows.extend(
                    {"base_case_id": base_case_id, "model": model_name, "seed": int(selected["seed"]), "learning_rate": selected_lr, "evaluation_split": "test", **row}
                    for row in _compute_per_band_metrics(np.load(target_file), np.load(pred_file), floor=config.relative_error_floor)
                )

    summary_rows: list[dict[str, Any]] = []
    for selection in selections:
        rows = [row for row in test_rows if row["base_case_id"] == selection["base_case_id"]]
        if len(rows) != 3:
            raise RuntimeError(f"Selected test aggregation for {selection['base_case_id']} does not have n=3.")
        summary = {**selection, "evaluation_split": "test", "n_seeds": len(rows)}
        for metric in ("avg_MRE_bands_2_10_pct", "mae", "rmse", "train_time_s", "eval_time_s", "inference_time_per_sample_ms", "throughput_samples_per_s", "parameter_count", "peak_gpu_memory_mb"):
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if len(values) == len(rows):
                summary[f"{metric}_mean"], summary[f"{metric}_std"] = _mean_std(values)
        summary_rows.append(summary)
    _write_rows_csv(run_dir / "validation_history.csv", validation_rows)
    _write_rows_csv(run_dir / "history.csv", test_rows)
    _write_rows_csv(run_dir / "leaderboard.csv", summary_rows)
    _write_rows_csv(run_dir / "per_band_metrics.csv", per_band_rows)
    _write_json(run_dir / "validation_selection_manifest.json", {"protocol": "validation_selected_lr_v1", "selections": selections})
    _write_json(run_dir / "integrity_summary.json", {"dataset_hash": dataset_hash, "split_signature": split_signature, "selection_split": "val", "test_split": "test", "all_summary_rows_n3": all(row["n_seeds"] == 3 for row in summary_rows), "test_driven_selection": False})
    _write_json(run_dir / "model_identity_summary.json", {"identity_rows": model_identity_rows})
    metrics = {"status": "ok", "stage": "forward_a1_fair_selection", "case_count": len(summary_rows), "wall_time_total_s": float(time.perf_counter() - wall_start), **collect_device_metadata(device_info, config.runtime)}
    write_metrics(run_dir, metrics)
    write_results(run_dir, {"status": "ok", "stage": "forward_a1_fair_selection", "neural_surrogate_no_mpb_meep": True, "test_driven_selection": False, "run_dir": str(run_dir)})
    _save_metric_figure(run_dir, metrics)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    logger.info("Finished validation-selected forward fair benchmark: %s", run_dir)
    return run_dir


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("case_id\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_for_csv(row.get(key)) for key in columns})


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{name} must be a mapping")


def _parse_benchmark_cases(config: ForwardRunConfig, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_cfg = _as_mapping(raw.get("model"), name="model")
    dataset_cfg = _as_mapping(raw.get("dataset"), name="dataset")
    model_default = config.model_name
    model_candidates = _as_sequence(model_cfg.get("candidates"), name="model.candidates", cast_str=True)
    case_matrix = _as_sequence(raw.get("benchmark"), name="benchmark") or _as_sequence(raw.get("cases"), name="cases")
    if case_matrix is None:
        input_repr_list = _as_sequence(
            dataset_cfg.get("input_representations"),
            name="dataset.input_representations",
            cast_str=True,
            include_single=True,
            fallback=[str(config.representation.input_repr)],
        )
        if not model_candidates:
            model_candidates = [model_default]
        cases: list[dict[str, Any]] = []
        for input_repr in input_repr_list:
            for model_name in model_candidates:
                cases.append(
                    {
                        "input_repr": input_repr,
                        "model": model_name,
                        "output_target": str(config.representation.output_target),
                        "model_params": dict(config.model_params),
                    }
                )
        return cases
    case_values: list[dict[str, Any]] = []
    for item in case_matrix:
        case = _as_mapping(item, name="benchmark item")
        case_values.append(
            {
                "input_repr": str(case.get("input_repr", config.representation.input_repr)),
                "model": str(case.get("model", model_default)),
                "output_target": str(case.get("output_target", config.representation.output_target)),
                "model_params": dict(config.model_params, **dict(case.get("model_params", {}))),
            }
        )
    return case_values


def _load_json_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_per_band_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    floor: float,
) -> list[dict[str, Any]]:
    if y_true.ndim < 3:
        return []
    band_metrics: list[dict[str, Any]] = []
    band_total = min(y_true.shape[-1], y_pred.shape[-1])
    for band in range(band_total):
        true_band = np.asarray(y_true, dtype=np.float64)[..., band]
        pred_band = np.asarray(y_pred, dtype=np.float64)[..., band]
        err = pred_band - true_band
        mse = float(np.mean(np.square(err)))
        mae = float(np.mean(np.abs(err)))
        band_metrics.append(
            {
                "band": band + 1,
                "mse": mse,
                "mae": mae,
                "rmse": float(math.sqrt(mse)),
                "avg_MRE_pct": _mre_pct(true_band, pred_band, floor=floor),
            }
        )
    return band_metrics


def _load_eval_model(
    config: ForwardRunConfig,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    torch: Any,
    device: Any,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[Any, bool]:
    model_name = config.model_name
    model_params = _filter_model_params(config.model_name, config.model_params)
    checkpoint_loaded = False
    if checkpoint_path is None and config.checkpoint_path is not None:
        checkpoint_path = config.checkpoint_path
    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_name = str(payload.get("model_name", model_name))
        model_params = _filter_model_params(model_name, {**model_params, **dict(payload.get("model_params", {}))})
        checkpoint_loaded = True
    model = build_model(
        model_name,
        input_shape=input_shape,
        output_shape=output_shape,
        **model_params,
    ).to(device)
    if checkpoint_loaded:
        model.load_state_dict(payload["state_dict"])
    return model, checkpoint_loaded


def _filter_model_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only model params that are declared in the registered spec defaults."""

    try:
        spec = get_model_spec(name)
    except KeyError:
        return dict(params)
    allowed = set(spec.defaults)
    return {key: value for key, value in params.items() if key in allowed}


def _write_benchmark_report(
    config: ForwardRunConfig,
    case_rows: list[dict[str, Any]],
    per_band_rows: list[dict[str, Any]],
    *,
    completed_at: datetime,
) -> Path:
    report_dir = Path("docs/results")
    report_path = Path(config.report_path) if config.report_path else report_dir / f"representation_benchmark_{completed_at.strftime('%Y%m%d')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best_mse = min(case_rows, key=lambda row: float(row.get("mse", float("inf"))) if row.get("mse") is not None else float("inf"))
    ranking_rows = case_rows[:]
    ranking_rows.sort(key=lambda row: float(row.get("mse", float("inf"))))
    content = [
        "# Representation Benchmark",
        "",
        f"- experiment_id: {config.experiment_id}",
        f"- completed_at: {completed_at.isoformat()}",
        f"- dataset: {config.dataset_path}",
        "",
        "## Case leaderboard",
        "",
        "| case_id | input_repr | model | mse | mae | rmse | avg_MRE_bands_2_10_pct |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ranking_rows:
        content.append(
            "| {case_id} | {input_repr} | {model} | {mse:.6f} | {mae:.6f} | {rmse:.6f} | {mre} |".format(
                case_id=row["case_id"],
                input_repr=row["input_repr"],
                model=row["model"],
                mse=float(row.get("mse", 0.0)),
                mae=float(row.get("mae", 0.0)),
                rmse=float(row.get("rmse", 0.0)),
                mre=(
                    "n/a"
                    if row.get("avg_MRE_bands_2_10_pct") is None
                    else f"{float(row['avg_MRE_bands_2_10_pct']):.6f}"
                ),
            )
        )
    content.extend(
        [
            "",
            f"Best by MSE: **{best_mse['case_id']}**",
            "",
            f"Per-band rows: {len(per_band_rows)}",
            "",
        ]
    )
    report_path.write_text("\n".join(content), encoding="utf-8")
    return report_path


def _as_sequence(
    value: Any,
    name: str,
    *,
    cast_str: bool = False,
    include_single: bool = False,
    fallback: list[str] | None = None,
) -> list[Any] | None:
    if value is None:
        if fallback is not None:
            return fallback
        return None
    if not isinstance(value, list):
        if include_single:
            return [str(value)]
        raise TypeError(f"{name} must be a list")
    values = []
    for item in value:
        if cast_str:
            if isinstance(item, str):
                values.append(item)
            else:
                values.append(str(item))
        else:
            values.append(item)
    return values


def _run_forward_train(
    config: ForwardRunConfig,
    *,
    run_dir_override: Path | None = None,
    torch_module: Any | None = None,
) -> Path:
    """Train a registered forward model from resolved config."""

    if _is_schema_smoke(config):
        return _run_forward_schema_smoke(config, run_dir_override=run_dir_override)

    wall_start = time.perf_counter()
    torch, DataLoader, TensorDataset = _require_torch() if torch_module is None else (torch_module, torch_module.utils.data.DataLoader, torch_module.utils.data.TensorDataset)
    _set_seed(config.seed, torch)
    device_info = _resolve_runtime_device(config, torch)
    device = device_info.torch_device
    reset_peak_gpu_memory(device, torch_module=torch)
    run_dir, logger, run_meta = _prepare_run(
        config,
        action="train",
        run_dir_override=run_dir_override,
        device_info=device_info,
    )
    logger.info("Starting local forward smoke train; this is not MPB/Meep validation.")

    setup_start = time.perf_counter()
    bundle, x, y = _load_arrays(config)
    model = build_model(
        config.model_name,
        input_shape=tuple(int(v) for v in x.shape[1:]),
        output_shape=tuple(int(v) for v in y.shape[1:]),
        **_filter_model_params(config.model_name, config.model_params),
    ).to(device)
    resume_checkpoint = _latest_epoch_checkpoint(run_dir)
    resumed_from_checkpoint = False
    resumed_checkpoint_path: Path | None = None
    resumed_checkpoint_epoch: int | None = None
    start_epoch = 1
    if resume_checkpoint is not None and not (run_dir / "model.pt").is_file():
        checkpoint_epoch, checkpoint_path = resume_checkpoint
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        resumed_from_checkpoint = True
        resumed_checkpoint_path = checkpoint_path
        resumed_checkpoint_epoch = int(payload.get("epoch", checkpoint_epoch))
        start_epoch = min(int(resumed_checkpoint_epoch) + 1, config.epochs + 1)
        logger.info(
            "Resuming forward train from %s at epoch %s",
            checkpoint_path,
            resumed_checkpoint_epoch,
        )
    optimizer = _make_optimizer(config, model, torch=torch)
    loss_fn = _make_forward_loss(config, torch=torch)
    scaler = create_grad_scaler(device_info, device, torch_module=torch)
    train_idx = _split_indices(bundle, "train")
    val_idx = _split_indices(bundle, "val")
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x[train_idx], dtype=torch.float32),
            torch.as_tensor(y[train_idx], dtype=torch.float32),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=device_info.num_workers,
        pin_memory=device_info.pin_memory,
    )
    setup_time_s = time.perf_counter() - setup_start

    history: list[dict[str, Any]] = []
    train_start = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = move_batch_to_device(
                (xb, yb),
                device,
                non_blocking=device_info.non_blocking_transfer,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device_info, device, torch_module=torch):
                loss = loss_fn(model(xb), yb)
            if bool(getattr(scaler, "is_enabled", lambda: False)()):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        val_metrics, _ = _evaluate_model(
            model=model,
            x=x,
            y=y,
            indices=val_idx,
            device=device,
            device_info=device_info,
            torch=torch,
            floor=config.relative_error_floor,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )
        if epoch in set(config.checkpoint_epochs):
            _save_forward_checkpoint(
                run_dir / f"checkpoint_epoch_{epoch}.pt",
                model=model,
                config=config,
                x=x,
                y=y,
                device_metadata=_device_metadata_with_peak(device_info, torch),
                epoch=epoch,
                torch=torch,
            )
    synchronize_device(device, torch_module=torch)
    train_time_s = time.perf_counter() - train_start

    eval_idx = _split_indices(bundle, config.eval_split)
    metrics, prediction = _evaluate_model(
        model=model,
        x=x,
        y=y,
        indices=eval_idx,
        device=device,
        device_info=device_info,
        torch=torch,
        floor=config.relative_error_floor,
    )
    device_metadata = _device_metadata_with_peak(device_info, torch)
    metrics.update(
        {
            "setup_time_s": float(setup_time_s),
            "train_time_s": float(train_time_s),
            "wall_time_total_s": float(time.perf_counter() - wall_start),
            "parameter_count": _parameter_count(model),
            "n_train": int(len(train_idx)),
            "n_eval": int(len(eval_idx)),
            "input_shape": list(x.shape[1:]),
            "target_shape": list(y.shape[1:]),
            "batch_size": int(config.batch_size),
            "effective_batch_size": int(config.batch_size),
            "resumed_from_checkpoint": bool(resumed_from_checkpoint),
            "resumed_checkpoint_epoch": resumed_checkpoint_epoch,
            "resumed_checkpoint_path": str(resumed_checkpoint_path) if resumed_checkpoint_path else None,
            **device_metadata,
        }
    )
    band_artifacts = save_forward_band_arrays(
        run_dir,
        target_bands=y[eval_idx],
        surrogate_pred_bands=prediction,
    )
    checkpoint_path = run_dir / "model.pt"
    _save_forward_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        x=x,
        y=y,
        device_metadata=device_metadata,
        epoch=config.epochs,
        torch=torch,
    )
    _write_history(run_dir / "history.csv", history)
    write_metrics(run_dir, metrics)
    write_results(
        run_dir,
        {
            "status": "ok",
            "stage": "forward_train",
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "resumed_from_checkpoint": bool(resumed_from_checkpoint),
            "resumed_checkpoint_epoch": resumed_checkpoint_epoch,
            "resumed_checkpoint_path": str(resumed_checkpoint_path) if resumed_checkpoint_path else None,
            "model_name": config.model_name,
            "dataset_path": str(config.dataset_path),
            "smoke_only_no_mpb_meep": True,
            "band_array_artifacts": band_artifacts,
            "metrics": {key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        },
    )
    _save_metric_figure(run_dir, metrics)
    logger.info("Finished forward smoke train: %s", run_dir)
    run_meta.update(device_metadata)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    return run_dir


def _run_forward_eval(
    config: ForwardRunConfig,
    *,
    run_dir_override: Path | None = None,
    torch_module: Any | None = None,
) -> Path:
    """Evaluate a registered forward model or checkpoint from resolved config."""

    if _is_schema_smoke(config):
        raise RuntimeError("Schema-only mode is for train-side smoke only in this repository stage.")

    wall_start = time.perf_counter()
    torch = _require_torch()[0] if torch_module is None else torch_module
    _set_seed(config.seed, torch)
    device_info = _resolve_runtime_device(config, torch)
    device = device_info.torch_device
    reset_peak_gpu_memory(device, torch_module=torch)
    discovered_checkpoint = _latest_checkpoint_for_config(config)
    if discovered_checkpoint is not None and config.checkpoint_path is None:
        config = replace(config, checkpoint_path=discovered_checkpoint)
    run_dir, logger, run_meta = _prepare_run(
        config,
        action="eval",
        run_dir_override=run_dir_override,
        device_info=device_info,
    )
    logger.info("Starting local forward smoke eval; this is not MPB/Meep validation.")

    setup_start = time.perf_counter()
    bundle, x, y = _load_arrays(config)
    model_name = config.model_name
    model_params = _filter_model_params(config.model_name, config.model_params)
    if config.checkpoint_path is not None:
        payload = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
        model_name = str(payload.get("model_name", model_name))
        model_params = _filter_model_params(model_name, {**dict(model_params), **dict(payload.get("model_params", {}))})
    model = build_model(
        model_name,
        input_shape=tuple(int(v) for v in x.shape[1:]),
        output_shape=tuple(int(v) for v in y.shape[1:]),
        **model_params,
    ).to(device)
    checkpoint_loaded = False
    if config.checkpoint_path is not None:
        payload = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        checkpoint_loaded = True
    setup_time_s = time.perf_counter() - setup_start

    eval_idx = _split_indices(bundle, config.eval_split)
    metrics, prediction = _evaluate_model(
        model=model,
        x=x,
        y=y,
        indices=eval_idx,
        device=device,
        device_info=device_info,
        torch=torch,
        floor=config.relative_error_floor,
    )
    device_metadata = _device_metadata_with_peak(device_info, torch)
    metrics.update(
        {
            "setup_time_s": float(setup_time_s),
            "wall_time_total_s": float(time.perf_counter() - wall_start),
            "parameter_count": _parameter_count(model),
            "n_eval": int(len(eval_idx)),
            "input_shape": list(x.shape[1:]),
            "target_shape": list(y.shape[1:]),
            "checkpoint_loaded": checkpoint_loaded,
            "batch_size": int(config.batch_size),
            "effective_batch_size": int(config.batch_size),
            **device_metadata,
        }
    )
    band_artifacts = save_forward_band_arrays(
        run_dir,
        target_bands=y[eval_idx],
        surrogate_pred_bands=prediction,
    )
    _write_history(
        run_dir / "history.csv",
        [
            {
                "split": config.eval_split,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        ],
    )
    write_metrics(run_dir, metrics)
    write_results(
        run_dir,
        {
            "status": "ok",
            "stage": "forward_eval",
            "run_dir": str(run_dir),
            "checkpoint_path": str(config.checkpoint_path) if config.checkpoint_path else None,
            "checkpoint_loaded": checkpoint_loaded,
            "model_name": model_name,
            "dataset_path": str(config.dataset_path),
            "smoke_only_no_mpb_meep": True,
            "band_array_artifacts": band_artifacts,
            "metrics": {key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        },
    )
    _save_metric_figure(run_dir, metrics)
    logger.info("Finished forward smoke eval: %s", run_dir)
    run_meta.update(device_metadata)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    return run_dir


__all__ = ["run_forward_benchmark", "run_forward_eval", "run_forward_train"]
