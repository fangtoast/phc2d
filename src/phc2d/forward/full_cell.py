"""Formal full-cell Full-Wh forward training on lazy HDF5 datasets.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.models import ModelSpec, build_model, describe_model_identity, get_model_spec
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.training import DeviceConfig, resolve_device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _nested_train_indices(
    idx_train: np.ndarray,
    sample_ids: Sequence[str],
    *,
    train_size: int,
    salt: str,
) -> np.ndarray:
    """Select a deterministic nested subset using stable sample-ID hashes."""

    indices = np.asarray(idx_train, dtype=np.int64)
    ids = [str(value) for value in sample_ids]
    if len(ids) != int(indices.size):
        raise ValueError("sample_ids must align one-to-one with idx_train")
    size = int(train_size)
    if size <= 0 or size > int(indices.size):
        raise ValueError(f"train_size must be in [1, {indices.size}], got {size}")
    ranked = sorted(
        zip(indices.tolist(), ids, strict=True),
        key=lambda item: (
            hashlib.sha256(f"{salt}|{item[1]}".encode("utf-8")).digest(),
            int(item[0]),
        ),
    )
    return np.sort(np.asarray([item[0] for item in ranked[:size]], dtype=np.int64))


def _sha256_checkpoint(path: Path) -> str:
    return _sha256_file(path)


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _epoch_100_continue_decision(
    history: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.005,
) -> dict[str, Any]:
    """Apply the preregistered epoch-100 validation-improvement gate."""

    by_epoch = {int(row["epoch"]): float(row["val_bands_2_10_mre_pct"]) for row in history}
    required = set(range(81, 101))
    if not required.issubset(by_epoch):
        raise ValueError("epoch-100 continuation gate requires finite epochs 81 through 100")
    first = np.asarray([by_epoch[epoch] for epoch in range(81, 91)], dtype=np.float64)
    second = np.asarray([by_epoch[epoch] for epoch in range(91, 101)], dtype=np.float64)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("epoch-100 continuation gate received non-finite validation values")
    mean_81_90 = float(np.mean(first))
    mean_91_100 = float(np.mean(second))
    if mean_81_90 <= 0.0:
        raise ValueError("epoch-100 continuation reference mean must be positive")
    relative_improvement = (mean_81_90 - mean_91_100) / mean_81_90
    return {
        "mean_epoch_81_90": mean_81_90,
        "mean_epoch_91_100": mean_91_100,
        "relative_improvement": relative_improvement,
        "threshold": float(threshold),
        "decision": "continue_to_max_200" if relative_improvement >= threshold else "stop_at_100",
    }


def _convergence_record(
    history: Sequence[Mapping[str, Any]],
    *,
    metric: str = "val_bands_2_10_mre_pct",
    first_window: tuple[int, int] = (181, 190),
    second_window: tuple[int, int] = (191, 200),
    threshold: float = 0.005,
) -> dict[str, Any]:
    """Evaluate the fixed terminal-window relative-change convergence gate."""

    by_epoch = {int(row["epoch"]): float(row[metric]) for row in history}
    required = set(range(first_window[0], first_window[1] + 1)) | set(range(second_window[0], second_window[1] + 1))
    if not required.issubset(by_epoch):
        missing = sorted(required.difference(by_epoch))
        raise ValueError(f"convergence gate requires complete terminal windows; missing epochs: {missing}")
    first = np.asarray([by_epoch[epoch] for epoch in range(first_window[0], first_window[1] + 1)], dtype=np.float64)
    second = np.asarray([by_epoch[epoch] for epoch in range(second_window[0], second_window[1] + 1)], dtype=np.float64)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("convergence gate received non-finite validation values")
    mean_first = float(np.mean(first))
    mean_second = float(np.mean(second))
    denominator = max(abs(mean_first), 1.0e-12)
    relative_change = abs(mean_second - mean_first) / denominator
    return {
        "metric": metric,
        "first_window": list(first_window),
        "second_window": list(second_window),
        "mean_first_window": mean_first,
        "mean_second_window": mean_second,
        "relative_change": relative_change,
        "threshold": float(threshold),
        "passed": bool(relative_change <= float(threshold)),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("Full-cell forward config must be a mapping")
    return dict(payload)


def _git_commit() -> str | None:
    deployed = os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT")
    if deployed:
        return deployed.strip()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


class _FullCellH5Dataset:
    """Worker-safe, read-only HDF5 sample adapter."""

    def __init__(
        self,
        path: Path,
        indices: np.ndarray,
        *,
        grid_field: str,
        layout: str,
        sampling: str,
        threshold: float | None,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.path = Path(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.grid_field = str(grid_field)
        self.layout = str(layout)
        self.sampling = str(sampling)
        self.threshold = threshold
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)
        self._handle: Any | None = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r")
        index = int(self.indices[int(position)])
        x = np.asarray(self._handle[self.grid_field][index], dtype=np.float32)
        if self.sampling == "deterministic_threshold":
            x = (x >= float(self.threshold)).astype(np.float32)
        if self.layout == "flat":
            x = x.reshape(-1, order="C")
        y = np.asarray(self._handle["Wh"][index], dtype=np.float32)
        y = (y - self.target_mean) / self.target_std
        return x, y

    def __del__(self) -> None:  # pragma: no cover - worker process lifecycle
        if self._handle is not None:
            self._handle.close()


def _validate_and_resolve(
    config: dict[str, Any],
    *,
    model_name: str,
    seed: int,
    admission_probe: bool,
    threshold_override: float | None = None,
    resolution_override: int | None = None,
    train_size_override: int | None = None,
    max_epochs_override: int | None = None,
    head_kind_override: str | None = None,
) -> dict[str, Any]:
    dataset = dict(config.get("dataset") or {})
    representation = dict(config.get("representation") or {})
    if threshold_override is not None:
        representation["threshold"] = float(threshold_override)
    if resolution_override is not None:
        if int(resolution_override) not in {32, 64, 128}:
            raise ValueError("resolution override must be 32, 64, or 128")
        representation["resolution"] = int(resolution_override)
        dataset["grid_field"] = f"material_weight_grid_{int(resolution_override)}"
    models = dict(config.get("models") or {})
    training = dict(config.get("training") or {})
    continuation = dict(config.get("continuation") or {})
    runtime = dict(config.get("runtime") or {})
    experiment = dict(config.get("experiment") or {})
    output = dict(config.get("output") or {})

    if model_name not in models:
        raise ValueError(f"model {model_name!r} is not configured under models")
    model_spec = get_model_spec(model_name)
    if model_spec.output_target != "Wh":
        raise ValueError(f"model {model_name!r} does not produce the Wh target")
    if int(seed) not in {0, 1, 2}:
        raise ValueError("formal full-cell forward seeds are fixed to 0, 1, 2")
    sampling = str(representation.get("sampling", "continuous"))
    threshold = representation.get("threshold")
    if sampling not in {"continuous", "deterministic_threshold"}:
        raise ValueError("representation.sampling must be continuous or deterministic_threshold")
    if sampling == "deterministic_threshold":
        if threshold not in {0.3, 0.4, 0.5, 0.6, 0.7}:
            raise ValueError("threshold must be one of 0.3, 0.4, 0.5, 0.6, 0.7")
        if bool(representation.get("physical_equivalence", True)):
            raise ValueError("threshold proxy inputs must set physical_equivalence=false")
    elif threshold is not None:
        raise ValueError("continuous representation requires threshold=null")

    layout = model_spec.input_layout
    learning_rates = dict(training.get("learning_rates") or {})
    if model_name not in learning_rates:
        raise ValueError(f"training.learning_rates is missing {model_name!r}")
    if train_size_override is not None:
        training["train_size"] = int(train_size_override)
    if max_epochs_override is not None:
        training["max_epochs"] = int(max_epochs_override)
    train_size = training.get("train_size")
    if train_size is not None and int(train_size) <= 0:
        raise ValueError("training.train_size must be positive")
    max_epochs = int(training.get("max_epochs", 100))
    if max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive")
    experiment_id = str(experiment.get("experiment_id", "full_cell_forward"))
    if sampling == "deterministic_threshold":
        experiment_id = f"{experiment_id}_t{int(round(float(threshold) * 10)):02d}"
    resolution = int(representation.get("resolution", 64))
    if resolution != 64:
        experiment_id = f"{experiment_id}_r{resolution}"
    if train_size is not None:
        experiment_id = f"{experiment_id}_n{int(train_size)}_e{max_epochs}"
    model_params = dict(models.get(model_name) or {})
    if head_kind_override is not None:
        if head_kind_override not in {"dense", "k_conditioned"}:
            raise ValueError("head kind override must be dense or k_conditioned")
        if not model_spec.accepts_k_points:
            raise ValueError(f"model {model_name!r} does not support selectable Full-Wh heads")
        model_params["head_kind"] = head_kind_override
    resolved = {
        "experiment": {
            **experiment,
            "experiment_id": experiment_id,
            "model": model_name,
            "seed": int(seed),
            "admission_probe": bool(admission_probe),
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
        },
        "dataset": dataset,
        "representation": {**representation, "layout": layout, "flatten_order": "row_major"},
        "model": {"name": model_name, "params": model_params},
        "training": {
            **training,
            "seed": int(seed),
            "learning_rate": float(learning_rates[model_name]),
            "max_epochs": 1 if admission_probe else max_epochs,
            "max_train_batches": 1 if admission_probe else None,
        },
        "runtime": runtime,
        "output": output,
        "continuation": continuation,
    }
    if not admission_probe:
        continuation_enabled = bool(continuation)
        allowed_epochs = {20, 100, 200, 600} if continuation_enabled else {20, 100, 200}
        if int(resolved["training"]["max_epochs"]) not in allowed_epochs:
            raise ValueError(f"formal training max_epochs must be one of {sorted(allowed_epochs)}")
        if int(training.get("early_stopping_patience", 0)) < 0:
            raise ValueError("early_stopping_patience must be nonnegative")
        if int(training.get("effective_batch_size", 0)) != 64:
            raise ValueError("formal training requires effective_batch_size=64")
        if str(training.get("optimizer", "")).lower() != "adamw":
            raise ValueError("formal training requires AdamW")
        if str(runtime.get("precision", "")).lower() != "fp32":
            raise ValueError("formal training requires FP32")
        if bool(runtime.get("amp_enabled")) or bool(runtime.get("tf32_enabled")):
            raise ValueError("formal training forbids AMP and TF32")
        if continuation_enabled:
            if str(experiment.get("experiment_id")) != "EXP-FWD-009":
                raise ValueError("weight-only convergence continuation requires EXP-FWD-009")
            if str(continuation.get("mode")) != "epoch200_weights_optimizer_reset":
                raise ValueError("unsupported continuation mode")
            if int(continuation.get("source_epoch", -1)) != 200:
                raise ValueError("continuation source_epoch must equal 200")
            if int(continuation.get("check_interval", -1)) != 20:
                raise ValueError("continuation check_interval must equal 20")
            if float(continuation.get("relative_threshold", -1.0)) != 0.005:
                raise ValueError("continuation relative_threshold must equal 0.005")
            if not bool(continuation.get("optimizer_reset", False)):
                raise ValueError("weight-only continuation must declare optimizer_reset=true")
            if int(resolved["training"]["max_epochs"]) != 600:
                raise ValueError("EXP-FWD-009 continuation max_epochs must equal 600")
            if training.get("required_stop_epoch") is not None:
                raise ValueError("continuation uses a convergence stop, not required_stop_epoch")
            if int(training.get("early_stopping_patience", -1)) != 0:
                raise ValueError("continuation requires early_stopping_patience=0")
        if training.get("required_stop_epoch") is not None:
            required_stop = int(training["required_stop_epoch"])
            if max_epochs != required_stop or required_stop != 200:
                raise ValueError("fixed-200 training requires max_epochs=required_stop_epoch=200")
            if int(training.get("early_stopping_patience", -1)) != 0:
                raise ValueError("fixed-200 training requires early_stopping_patience=0")
            if bool(training.get("epoch_100_continue_gate", False)):
                raise ValueError("fixed-200 training forbids epoch_100_continue_gate")
            if not bool(training.get("require_terminal_checkpoint", False)):
                raise ValueError("fixed-200 training requires a terminal checkpoint")
            if tuple(int(value) for value in training.get("convergence_window", [])) != (181, 200):
                raise ValueError("fixed-200 training requires convergence_window=[181, 200]")
            if str(training.get("convergence_metric", "")) != "validation_bands_2_10_mre":
                raise ValueError("fixed-200 training requires validation_bands_2_10_mre")
            if float(training.get("convergence_relative_threshold", -1.0)) != 0.005:
                raise ValueError("fixed-200 convergence threshold must be 0.005")
        gate_enabled = bool(training.get("epoch_100_continue_gate", False))
        if gate_enabled:
            if int(resolved["training"]["max_epochs"]) != 200:
                raise ValueError("epoch-100 continuation gate requires max_epochs=200")
            threshold = float(training.get("continue_gate_min_relative_improvement", 0.005))
            if threshold != 0.005:
                raise ValueError("preregistered continuation threshold must equal 0.005")
            if int(training.get("early_stopping_patience", 0)) != 15:
                raise ValueError("epoch-100 continuation requires early_stopping_patience=15")
    return resolved


def _read_contract(resolved: Mapping[str, Any]) -> dict[str, Any]:
    import h5py

    dataset = dict(resolved["dataset"])
    representation = dict(resolved["representation"])
    training = dict(resolved["training"])
    path = Path(str(dataset["path"]))
    grid_field = str(dataset.get("grid_field", "material_weight_grid_64"))
    with h5py.File(path, "r") as handle:
        required = {grid_field, "Wh", "idx_train", "idx_val", "idx_test"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"full-cell HDF5 missing required fields: {missing}")
        idx_train = np.asarray(handle["idx_train"], dtype=np.int64)
        idx_val = np.asarray(handle["idx_val"], dtype=np.int64)
        idx_test = np.asarray(handle["idx_test"], dtype=np.int64)
        sorted_train = np.sort(idx_train)
        total_samples = int(handle["Wh"].shape[0])
        train_size = int(training.get("train_size", sorted_train.size))
        subset_salt = str(training.get("train_subset_salt", "phc2d-full-cell-50k-v1"))
        full_dataset_final_fit = False
        if train_size < int(sorted_train.size):
            if "sample_id" not in handle:
                raise KeyError("nested train subsets require the HDF5 sample_id field")
            sample_ids = np.asarray(handle["sample_id"][sorted_train]).astype("U").tolist()
            selected_train = _nested_train_indices(
                sorted_train,
                sample_ids,
                train_size=train_size,
                salt=subset_salt,
            )
        elif train_size == int(sorted_train.size):
            selected_train = sorted_train
        elif train_size == total_samples:
            selected_train = np.arange(total_samples, dtype=np.int64)
            full_dataset_final_fit = True
        else:
            raise ValueError(
                f"training.train_size={train_size} must not exceed idx_train size "
                f"{sorted_train.size} unless it equals the full dataset size {total_samples}"
            )
        grid_shape = tuple(int(v) for v in handle[grid_field].shape[1:])
        target_shape = tuple(int(v) for v in handle["Wh"].shape[1:])
        y_train = np.asarray(handle["Wh"][selected_train], dtype=np.float32)
        k_points = np.asarray(handle["k_points"], dtype=np.float32) if "k_points" in handle else None
        schema_version = str(handle.attrs.get("schema_version", ""))
    if len(grid_shape) != 2 or len(target_shape) != 2:
        raise ValueError(f"unexpected grid/target shapes: {grid_shape}, {target_shape}")
    if k_points is not None and k_points.shape != (target_shape[0], 2):
        raise ValueError(
            f"k_points shape {tuple(k_points.shape)} does not match target k-axis {target_shape[0]}"
        )
    expected_resolution = int(representation.get("resolution", grid_shape[0]))
    if grid_shape != (expected_resolution, expected_resolution):
        raise ValueError(f"grid field shape {grid_shape} does not match resolution {expected_resolution}")
    hashes = {
        "dataset_sha256": str(dataset.get("expected_dataset_sha256") or _sha256_file(path)),
        "idx_train_hash": _sha256_array(idx_train),
        "idx_val_hash": _sha256_array(idx_val),
        "idx_test_hash": _sha256_array(idx_test),
    }
    for key, actual in hashes.items():
        expected = dataset.get(f"expected_{key}")
        if expected and str(expected) != actual:
            raise ValueError(f"{key} mismatch: {actual} != {expected}")
    target_mean = np.mean(y_train, axis=0, dtype=np.float64).astype(np.float32)
    target_std = np.std(y_train, axis=0, dtype=np.float64).astype(np.float32)
    target_std = np.maximum(target_std, np.float32(1.0e-6))
    return {
        "path": path,
        "grid_field": grid_field,
        "grid_shape": grid_shape,
        "target_shape": target_shape,
        "schema_version": schema_version,
        "idx_train": idx_train,
        "selected_train": selected_train,
        "train_subset_hash": _sha256_array(selected_train),
        "train_subset_salt": subset_salt,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "full_dataset_final_fit": full_dataset_final_fit,
        "validation_overlap_count": int(np.intersect1d(selected_train, idx_val).size),
        "test_overlap_count": int(np.intersect1d(selected_train, idx_test).size),
        "target_mean": target_mean,
        "target_std": target_std,
        "k_points": k_points,
        "k_points_sha256": (
            hashlib.sha256(np.ascontiguousarray(k_points).tobytes()).hexdigest()
            if k_points is not None
            else None
        ),
        **hashes,
    }


def _lane_index(lane_name: Any) -> int:
    normalized = str(lane_name or "").strip().lower()
    if "binary" in normalized:
        return 0
    if "graded" in normalized or "continuous" in normalized:
        return 1
    raise ValueError(f"cannot map experiment lane {lane_name!r} to binary=0 or graded=1")


def _forward_model(
    model: Any,
    xb: Any,
    *,
    model_spec: ModelSpec,
    k_points: Any,
    lane_index: int | None,
    torch: Any,
) -> Any:
    kwargs: dict[str, Any] = {}
    if model_spec.accepts_k_points:
        kwargs["k_points"] = k_points
    if model_spec.accepts_lane_id:
        if lane_index is None:
            raise ValueError(f"model {model_spec.name!r} requires a lane id")
        kwargs["lane_id"] = torch.full(
            (int(xb.shape[0]),), lane_index, dtype=torch.long, device=xb.device
        )
    return model(xb, **kwargs)


def _evaluate(
    model: Any,
    loader: Any,
    *,
    device: Any,
    mean: Any,
    std: Any,
    model_spec: ModelSpec,
    k_points: Any,
    lane_index: int | None,
    torch: Any,
) -> dict[str, float]:
    model.eval()
    squared = 0.0
    absolute = 0.0
    relative = 0.0
    relative_band1 = 0.0
    relative_all = 0.0
    normalized_squared = 0.0
    count = 0
    inference_start = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            prediction_norm = _forward_model(
                model,
                xb,
                model_spec=model_spec,
                k_points=k_points,
                lane_index=lane_index,
                torch=torch,
            )
            if not bool(torch.isfinite(prediction_norm).all().item()):
                raise RuntimeError("non-finite validation prediction")
            normalized_diff = prediction_norm - yb
            normalized_squared += float(torch.sum(normalized_diff * normalized_diff).detach().cpu())
            prediction = prediction_norm * std + mean
            target = yb * std + mean
            diff = prediction - target
            squared += float(torch.sum(diff * diff).detach().cpu())
            absolute += float(torch.sum(torch.abs(diff)).detach().cpu())
            relative += float(
                torch.sum(torch.abs(diff[..., 1:]) / torch.clamp(torch.abs(target[..., 1:]), min=1.0e-8)).detach().cpu()
            )
            relative_band1 += float(
                torch.sum(torch.abs(diff[..., :1]) / torch.clamp(torch.abs(target[..., :1]), min=1.0e-8)).detach().cpu()
            )
            relative_all += float(
                torch.sum(torch.abs(diff) / torch.clamp(torch.abs(target), min=1.0e-8)).detach().cpu()
            )
            count += int(target.numel())
    elapsed = time.perf_counter() - inference_start
    band_count = count * max(int(mean.shape[-1]) - 1, 0) / max(int(mean.shape[-1]), 1)
    sample_count = max(count // int(mean.shape[-1]), 1)
    return {
        "mse": squared / max(count, 1),
        "normalized_mse": normalized_squared / max(count, 1),
        "mae": absolute / max(count, 1),
        "rmse": float(np.sqrt(squared / max(count, 1))),
        "bands_2_10_mre_pct": 100.0 * relative / max(band_count, 1),
        "band1_mre_pct": 100.0 * relative_band1 / sample_count,
        "bands_1_10_mre_pct": 100.0 * relative_all / max(count, 1),
        "eval_time_s": elapsed,
    }


def _plot_history(
    run_dir: Path,
    history: list[Mapping[str, Any]],
    *,
    monitor_label: str = "validation",
) -> None:
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train normalized MSE")
    axes[0].plot(
        epochs,
        [float(row["val_loss"]) for row in history],
        label=f"{monitor_label} normalized MSE",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [float(row["val_bands_2_10_mre_pct"]) for row in history])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(f"{monitor_label} bands 2-10 MRE (%)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(run_dir / "figures" / f"training_curve.{suffix}", dpi=180)
    plt.close(fig)


def run_full_cell_forward(
    config_path: str | Path,
    *,
    model_name: str,
    seed: int,
    run_dir: str | Path | None = None,
    admission_probe: bool = False,
    threshold: float | None = None,
    resolution: int | None = None,
    train_size: int | None = None,
    max_epochs: int | None = None,
    head_kind: str | None = None,
    warm_start_run: str | Path | None = None,
) -> Path:
    """Run one isolated model-plus-seed job from a frozen wave config."""

    config_file = Path(config_path)
    raw_config = _load_yaml(config_file)
    if warm_start_run is not None:
        raw_config.setdefault("continuation", {})["source_run"] = str(Path(warm_start_run).resolve())
    resolved = _validate_and_resolve(
        raw_config,
        model_name=model_name,
        seed=seed,
        admission_probe=admission_probe,
        threshold_override=threshold,
        resolution_override=resolution,
        train_size_override=train_size,
        max_epochs_override=max_epochs,
        head_kind_override=head_kind,
    )
    experiment = dict(resolved["experiment"])
    representation = dict(resolved["representation"])
    training = dict(resolved["training"])
    runtime = dict(resolved["runtime"])
    output = dict(resolved["output"])
    task = str(experiment["experiment_id"])
    out = make_run_dir(task, runs_dir=output.get("runs_dir", "runs"), run_dir=run_dir)
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "python -m phc2d.forward.full_cell",
            "config_path": str(config_file),
            "model": model_name,
            "seed": int(seed),
            "validation_boundary": "12_job_admission_probe" if admission_probe else "formal_full_cell_forward_training",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)
    wall_start = time.perf_counter()

    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_cfg = DeviceConfig.from_mapping(runtime)
    device_info = resolve_device(device_cfg, torch_module=torch)
    device = device_info.torch_device
    contract = _read_contract(resolved)
    final_fit = bool(contract["full_dataset_final_fit"])
    layout = str(representation["layout"])
    model_spec = get_model_spec(model_name)
    input_shape = contract["grid_shape"] if layout == "image" else (int(np.prod(contract["grid_shape"])),)
    model_kwargs = dict(resolved["model"]["params"])
    if model_spec.accepts_k_points:
        model_kwargs["k_points"] = contract["k_points"]
    model = build_model(
        model_name,
        input_shape=input_shape,
        output_shape=contract["target_shape"],
        **model_kwargs,
    ).to(device)
    warm_start: dict[str, Any] | None = None
    if resolved.get("continuation"):
        from phc2d.forward.full_cell_warm_start import load_weight_only_warm_start

        source_run = resolved["continuation"].get("source_run")
        if not source_run:
            raise ValueError("continuation source_run is required")
        warm_start = load_weight_only_warm_start(
            source_run,
            expected_model=model_name,
            expected_seed=seed,
            torch_module=torch,
        )
        model.load_state_dict(warm_start["terminal_payload"]["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1.0e-5)),
    )
    loss_fn = torch.nn.MSELoss()
    dataset_kwargs = {
        "path": contract["path"],
        "grid_field": contract["grid_field"],
        "layout": layout,
        "sampling": representation.get("sampling", "continuous"),
        "threshold": representation.get("threshold"),
        "target_mean": contract["target_mean"],
        "target_std": contract["target_std"],
    }
    train_dataset = _FullCellH5Dataset(indices=contract["selected_train"], **dataset_kwargs)
    val_dataset = _FullCellH5Dataset(indices=contract["idx_val"], **dataset_kwargs)
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(training.get("micro_batch_size", 64)),
        "num_workers": int(device_info.num_workers),
        "pin_memory": bool(device_info.pin_memory),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    mean_t = torch.as_tensor(contract["target_mean"], dtype=torch.float32, device=device)
    std_t = torch.as_tensor(contract["target_std"], dtype=torch.float32, device=device)
    k_points_t = (
        torch.as_tensor(contract["k_points"], dtype=torch.float32, device=device)
        if model_spec.accepts_k_points and contract["k_points"] is not None
        else None
    )
    lane_index = _lane_index(experiment.get("lane")) if model_spec.accepts_lane_id else None
    accumulation = int(training.get("effective_batch_size", 64)) // int(training.get("micro_batch_size", 64))
    if accumulation < 1 or int(training.get("effective_batch_size", 64)) % int(training.get("micro_batch_size", 64)):
        raise ValueError("effective_batch_size must be an integer multiple of micro_batch_size")

    history: list[dict[str, Any]] = [] if warm_start is None else list(warm_start["source_history"])
    best_mre = float("inf") if warm_start is None else min(float(row["val_bands_2_10_mre_pct"]) for row in history)
    best_epoch = 0 if warm_start is None else int(warm_start["source_result"]["best_epoch"])
    stale_epochs = 0
    train_start = time.perf_counter()
    max_batches = training.get("max_train_batches")
    optimizer_steps = 0 if warm_start is None else int(history[-1]["optimizer_steps"])
    samples_seen = 0 if warm_start is None else int(history[-1]["samples_seen"])
    wall_time_offset = 0.0 if warm_start is None else float(history[-1]["wall_time_s"])
    continuation_convergence: dict[str, Any] | None = None
    if warm_start is not None:
        from phc2d.forward.full_cell_warm_start import seed_derivative_best_checkpoint

        seed_derivative_best_checkpoint(warm_start, out)
    gate_enabled = bool(training.get("epoch_100_continue_gate", False)) and not admission_probe
    gate_record: dict[str, Any] | None = None
    epoch100_snapshot: Path | None = None
    start_epoch = 1 if warm_start is None else int(warm_start["source_epoch"]) + 1
    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_count = 0
        for batch_index, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            prediction = _forward_model(
                model,
                xb,
                model_spec=model_spec,
                k_points=k_points_t,
                lane_index=lane_index,
                torch=torch,
            )
            loss = loss_fn(prediction, yb)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}, batch {batch_index}")
            (loss / accumulation).backward()
            if bool(training.get("require_finite_gradients", admission_probe)) and any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite training gradient at epoch {epoch}, batch {batch_index}")
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            train_sum += float(loss.detach().cpu()) * int(xb.shape[0])
            train_count += int(xb.shape[0])
            samples_seen += int(xb.shape[0])
            if max_batches is not None and batch_index >= int(max_batches):
                break
        val = _evaluate(
            model,
            val_loader,
            device=device,
            mean=mean_t,
            std=std_t,
            model_spec=model_spec,
            k_points=k_points_t,
            lane_index=lane_index,
            torch=torch,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_loss": val["normalized_mse"],
            "val_band1_mre_pct": val["band1_mre_pct"],
            "val_bands_2_10_mre_pct": val["bands_2_10_mre_pct"],
            "val_bands_1_10_mre_pct": val["bands_1_10_mre_pct"],
            "learning_rate": float(training["learning_rate"]),
            "epoch_time_s": time.perf_counter() - epoch_start,
            "wall_time_s": wall_time_offset + time.perf_counter() - wall_start,
            "optimizer_steps": optimizer_steps,
            "samples_seen": samples_seen,
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6g val_mre=%.6g%% epoch_time_s=%.3f",
            epoch,
            row["train_loss"],
            row["val_bands_2_10_mre_pct"],
            row["epoch_time_s"],
        )
        improved = float(val["bands_2_10_mre_pct"]) < best_mre
        if improved:
            best_mre = float(val["bands_2_10_mre_pct"])
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        write_checkpoint = (not final_fit and improved) or (
            final_fit and epoch == int(training["max_epochs"])
        )
        if write_checkpoint:
            checkpoint_tmp = out / "model.pt.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_params": dict(resolved["model"]["params"]),
                    "model_identity": describe_model_identity(
                        model, model_name, model_spec, model_kwargs=resolved["model"]["params"]
                    ),
                    "input_shape": input_shape,
                    "output_shape": contract["target_shape"],
                    "k_points": (
                        torch.as_tensor(contract["k_points"], dtype=torch.float32)
                        if contract["k_points"] is not None
                        else None
                    ),
                    "k_points_sha256": contract["k_points_sha256"],
                    "lane_index": lane_index,
                    "target_mean": torch.as_tensor(contract["target_mean"], dtype=torch.float32),
                    "target_std": torch.as_tensor(contract["target_std"], dtype=torch.float32),
                    "epoch": epoch,
                    "selection_split": "none_final_fit" if final_fit else "validation",
                    "validation_bands_2_10_mre_pct": None if final_fit else best_mre,
                    "training_overlap_monitor_bands_2_10_mre_pct": (
                        float(val["bands_2_10_mre_pct"]) if final_fit else None
                    ),
                    "resolved_config": resolved,
                    "warm_start": None if warm_start is None else {
                        key: value for key, value in warm_start.items()
                        if key not in {"source_history", "source_result", "terminal_payload"}
                    },
                },
                checkpoint_tmp,
            )
            os.replace(checkpoint_tmp, out / "model.pt")
        _write_history(out / "history.csv", history)
        if warm_start is not None and epoch % int(resolved["continuation"]["check_interval"]) == 0:
            from phc2d.forward.full_cell_warm_start import rolling_convergence_record

            continuation_convergence = rolling_convergence_record(
                history,
                end_epoch=epoch,
                metric="val_bands_2_10_mre_pct",
                threshold=float(resolved["continuation"]["relative_threshold"]),
            )
            logger.info(
                "continuation convergence epoch=%d relative_change=%.8f passed=%s",
                epoch,
                continuation_convergence["relative_change"],
                continuation_convergence["passed"],
            )
            if bool(continuation_convergence["passed"]):
                break
        if gate_enabled and epoch == 100:
            gate_record = _epoch_100_continue_decision(
                history,
                threshold=float(training.get("continue_gate_min_relative_improvement", 0.005)),
            )
            epoch100_snapshot = out / "epoch100.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": 100,
                    "checkpoint_role": "epoch_100_gate_snapshot",
                    "gate_record": gate_record,
                },
                epoch100_snapshot,
            )
            logger.info(
                "epoch-100 gate relative_improvement=%.8f threshold=%.8f decision=%s",
                gate_record["relative_improvement"],
                gate_record["threshold"],
                gate_record["decision"],
            )
            if gate_record["decision"] == "stop_at_100":
                break
            stale_epochs = 0
        patience = int(training.get("early_stopping_patience", 15))
        patience_active = not gate_enabled or (epoch > 100 and gate_record is not None)
        if not final_fit and not admission_probe and patience_active and patience > 0 and stale_epochs >= patience:
            logger.info("early stopping at epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    train_time = time.perf_counter() - train_start
    stop_epoch = int(history[-1]["epoch"])
    stop_snapshot = epoch100_snapshot
    if gate_enabled and stop_epoch > 100:
        stop_snapshot = out / "continuation_stop.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": stop_epoch,
                "checkpoint_role": "post_gate_actual_stop_snapshot",
                "gate_record": gate_record,
            },
            stop_snapshot,
        )
    required_stop_epoch = training.get("required_stop_epoch")
    convergence: dict[str, Any] | None = None
    terminal_checkpoint: Path | None = None
    if required_stop_epoch is not None and not admission_probe:
        required_stop_epoch = int(required_stop_epoch)
        if stop_epoch != required_stop_epoch:
            raise RuntimeError(f"required terminal epoch {required_stop_epoch}, observed {stop_epoch}")
        convergence_window = tuple(int(value) for value in training.get("convergence_window", [181, 200]))
        if convergence_window != (181, 200):
            raise ValueError("EXP-FWD-008 convergence_window must be [181, 200]")
        convergence = _convergence_record(
            history,
            metric=("val_bands_2_10_mre_pct" if str(training.get("convergence_metric")) == "validation_bands_2_10_mre" else str(training.get("convergence_metric", "val_bands_2_10_mre_pct"))),
            first_window=(181, 190),
            second_window=(191, 200),
            threshold=float(training.get("convergence_relative_threshold", 0.005)),
        )
        convergence["contract_metric"] = str(training.get("convergence_metric", "validation_bands_2_10_mre"))
        terminal_checkpoint = out / f"epoch{required_stop_epoch}.pt"
        terminal_tmp = terminal_checkpoint.with_suffix(terminal_checkpoint.suffix + ".tmp")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "model_params": dict(resolved["model"]["params"]),
                "model_identity": describe_model_identity(
                    model, model_name, model_spec, model_kwargs=resolved["model"]["params"]
                ),
                "input_shape": input_shape,
                "output_shape": contract["target_shape"],
                "k_points": torch.as_tensor(contract["k_points"], dtype=torch.float32) if contract["k_points"] is not None else None,
                "k_points_sha256": contract["k_points_sha256"],
                "lane_index": lane_index,
                "target_mean": torch.as_tensor(contract["target_mean"], dtype=torch.float32),
                "target_std": torch.as_tensor(contract["target_std"], dtype=torch.float32),
                "epoch": required_stop_epoch,
                "checkpoint_role": "terminal_epoch",
                "selection_split": "terminal_training_state",
                "convergence": convergence,
                "resolved_config": resolved,
            },
            terminal_tmp,
        )
        os.replace(terminal_tmp, terminal_checkpoint)
    elif warm_start is not None and not admission_probe:
        convergence = continuation_convergence
        if convergence is None:
            raise RuntimeError("continuation ended without a convergence checkpoint")
        terminal_checkpoint = out / f"epoch{stop_epoch}.pt"
        terminal_tmp = terminal_checkpoint.with_suffix(terminal_checkpoint.suffix + ".tmp")
        from phc2d.forward.full_cell_warm_start import public_warm_start_metadata

        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "model_params": dict(resolved["model"]["params"]),
                "model_identity": describe_model_identity(model, model_name, model_spec, model_kwargs=resolved["model"]["params"]),
                "input_shape": input_shape,
                "output_shape": contract["target_shape"],
                "k_points": torch.as_tensor(contract["k_points"], dtype=torch.float32) if contract["k_points"] is not None else None,
                "k_points_sha256": contract["k_points_sha256"],
                "lane_index": lane_index,
                "target_mean": torch.as_tensor(contract["target_mean"], dtype=torch.float32),
                "target_std": torch.as_tensor(contract["target_std"], dtype=torch.float32),
                "epoch": stop_epoch,
                "checkpoint_role": "continuation_terminal_epoch",
                "selection_split": "terminal_training_state",
                "convergence": convergence,
                "warm_start": public_warm_start_metadata(warm_start),
                "resolved_config": resolved,
            },
            terminal_tmp,
        )
        os.replace(terminal_tmp, terminal_checkpoint)
    checkpoint = out / "model.pt"
    if not checkpoint.is_file():
        role = "final-epoch final-fit" if final_fit else "best-validation"
        raise RuntimeError(f"{role} checkpoint was not written")
    try:
        checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 2.1 supports the legacy default only on some builds.
        checkpoint_payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    final_val = _evaluate(
        model,
        val_loader,
        device=device,
        mean=mean_t,
        std=std_t,
        model_spec=model_spec,
        k_points=k_points_t,
        lane_index=lane_index,
        torch=torch,
    )
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device_info.is_cuda else 0.0
    checkpoint_epoch = int(checkpoint_payload["epoch"])
    metrics = {
        **final_val,
        "best_validation_bands_2_10_mre_pct": None if final_fit else best_mre,
        "best_training_overlap_monitor_mre_pct": best_mre if final_fit else None,
        "monitor_is_unbiased_validation": not final_fit,
        "best_epoch": checkpoint_epoch if final_fit else best_epoch,
        "best_monitor_epoch": best_epoch,
        "checkpoint_epoch": checkpoint_epoch,
        "stop_epoch": stop_epoch,
        "epoch_100_continue_gate": gate_record,
        "train_time_s": train_time,
        "wall_time_total_s": time.perf_counter() - wall_start,
        "inference_time_per_sample_ms": 1000.0 * final_val["eval_time_s"] / max(len(val_dataset), 1),
        "throughput_samples_per_s": len(val_dataset) / max(final_val["eval_time_s"], 1.0e-12),
        "peak_gpu_memory_mb": peak_mb,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "effective_batch_size": int(training.get("effective_batch_size", 64)),
        "micro_batch_size": int(training.get("micro_batch_size", 64)),
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "train_subset_hash": contract["train_subset_hash"],
        "model_identity": describe_model_identity(
            model, model_name, model_spec, model_kwargs=resolved["model"]["params"]
        ),
        "convergence": convergence,
        "required_stop_epoch": required_stop_epoch,
        "terminal_checkpoint_path": str(terminal_checkpoint) if terminal_checkpoint else None,
        "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
        "history_epoch_count": len(history),
        "warm_start": None if warm_start is None else public_warm_start_metadata(warm_start),
    }
    _plot_history(
        out,
        history,
        monitor_label="training-overlap monitor" if final_fit else "validation",
    )
    write_metrics(out, metrics)
    (out / "model_summary.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "parameter_count": metrics["parameter_count"],
                "input_shape": list(input_shape),
                "output_shape": list(contract["target_shape"]),
                "best_validation_checkpoint": str(checkpoint),
                "best_validation_checkpoint_sha256": _sha256_checkpoint(checkpoint),
                "terminal_checkpoint": str(terminal_checkpoint) if terminal_checkpoint else None,
                "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
                "model_identity": metrics["model_identity"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = (
        "full_cell_forward_admission_probe_pass"
        if admission_probe
        else "continuation_training_success"
        if warm_start is not None
        else "formal_training_success"
    )
    if not admission_probe and convergence is not None and not bool(convergence["passed"]):
        status = "training_complete_convergence_not_met"
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
            "selection_split": "none_final_fit" if final_fit else "validation",
            "checkpoint_role": "final_epoch_final_fit" if final_fit else "best_validation",
            "validation_overlap": final_fit,
            "validation_metrics_admissible": not final_fit,
            "test_executed": False,
            "model": model_name,
            "model_identity": metrics["model_identity"],
            "seed": int(seed),
            "lane": experiment.get("lane"),
            "representation": representation,
            "train_size": len(train_dataset),
            "max_epochs": int(training["max_epochs"]),
            "train_subset_hash": contract["train_subset_hash"],
            "best_epoch": checkpoint_epoch if final_fit else best_epoch,
            "stop_epoch": stop_epoch,
            "best_monitor_epoch": best_epoch,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "terminal_checkpoint_path": str(terminal_checkpoint) if terminal_checkpoint else None,
            "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
            "required_stop_epoch": required_stop_epoch,
            "convergence": convergence,
            "history_epoch_count": len(history),
            "warm_start": None if warm_start is None else public_warm_start_metadata(warm_start),
            "epoch_100_continue_gate": gate_record,
            "epoch_100_checkpoint_path": str(epoch100_snapshot) if epoch100_snapshot else None,
            "epoch_100_checkpoint_sha256": _sha256_checkpoint(epoch100_snapshot) if epoch100_snapshot else None,
            "actual_stop_checkpoint_path": str(stop_snapshot) if stop_snapshot else None,
            "actual_stop_checkpoint_sha256": _sha256_checkpoint(stop_snapshot) if stop_snapshot else None,
            "next_gate": (
                "wave_admission_decision"
                if admission_probe
                else "continuation_checkpoint_freeze"
                if warm_start is not None
                else "12_job_final_fit_aggregation"
                if final_fit
                else "12_job_wave_aggregation"
            ),
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.forward.full_cell",
            "config_path": str(config_file),
            "deployed_git_commit": _git_commit(),
            "dataset_path": str(contract["path"]),
            "dataset_sha256": contract["dataset_sha256"],
            "idx_train_hash": contract["idx_train_hash"],
            "train_subset_hash": contract["train_subset_hash"],
            "train_subset_salt": contract["train_subset_salt"],
            "idx_val_hash": contract["idx_val_hash"],
            "idx_test_hash": contract["idx_test_hash"],
            "k_points_sha256": contract["k_points_sha256"],
            "full_dataset_final_fit": final_fit,
            "validation_overlap_count": contract["validation_overlap_count"],
            "test_overlap_count": contract["test_overlap_count"],
            "schema_version": contract["schema_version"],
            "model": model_name,
            "seed": int(seed),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
            **device_info.metadata(peak_gpu_memory_mb=peak_mb),
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete run artifact contract: {missing}")
    logger.info("terminal_status=%s run_dir=%s", status, out)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--admission-probe", action="store_true")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--head-kind", choices=("dense", "k_conditioned"))
    parser.add_argument("--warm-start-run", type=Path)
    args = parser.parse_args(argv)
    print(
        run_full_cell_forward(
            args.config,
            model_name=args.model,
            seed=args.seed,
            run_dir=args.run_dir,
            admission_probe=args.admission_probe,
            threshold=args.threshold,
            resolution=args.resolution,
            train_size=args.train_size,
            max_epochs=args.max_epochs,
            head_kind=args.head_kind,
            warm_start_run=args.warm_start_run,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["_epoch_100_continue_decision", "_nested_train_indices", "run_full_cell_forward"]
