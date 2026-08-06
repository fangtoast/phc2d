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

from phc2d.models import build_model
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
    runtime = dict(config.get("runtime") or {})
    experiment = dict(config.get("experiment") or {})
    output = dict(config.get("output") or {})

    required_models = {"mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2"}
    if model_name not in required_models:
        raise ValueError(f"model must be one of {sorted(required_models)}")
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

    layout = "flat" if model_name in {"mlp", "res_mlp_wide"} else "image"
    learning_rates = dict(training.get("learning_rates") or {})
    experiment_id = str(experiment.get("experiment_id", "full_cell_forward"))
    if sampling == "deterministic_threshold":
        experiment_id = f"{experiment_id}_t{int(round(float(threshold) * 10)):02d}"
    resolution = int(representation.get("resolution", 64))
    if resolution != 64:
        experiment_id = f"{experiment_id}_r{resolution}"
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
        "model": {"name": model_name, "params": dict(models.get(model_name) or {})},
        "training": {
            **training,
            "seed": int(seed),
            "learning_rate": float(learning_rates[model_name]),
            "max_epochs": 1 if admission_probe else int(training.get("max_epochs", 100)),
            "max_train_batches": 1 if admission_probe else None,
        },
        "runtime": runtime,
        "output": output,
    }
    if not admission_probe:
        if int(resolved["training"]["max_epochs"]) != 100:
            raise ValueError("formal training requires exactly 100 maximum epochs")
        if int(training.get("early_stopping_patience", 0)) != 15:
            raise ValueError("formal training requires early_stopping_patience=15")
        if int(training.get("effective_batch_size", 0)) != 64:
            raise ValueError("formal training requires effective_batch_size=64")
        if str(training.get("optimizer", "")).lower() != "adamw":
            raise ValueError("formal training requires AdamW")
        if str(runtime.get("precision", "")).lower() != "fp32":
            raise ValueError("formal training requires FP32")
        if bool(runtime.get("amp_enabled")) or bool(runtime.get("tf32_enabled")):
            raise ValueError("formal training forbids AMP and TF32")
    return resolved


def _read_contract(resolved: Mapping[str, Any]) -> dict[str, Any]:
    import h5py

    dataset = dict(resolved["dataset"])
    representation = dict(resolved["representation"])
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
        grid_shape = tuple(int(v) for v in handle[grid_field].shape[1:])
        target_shape = tuple(int(v) for v in handle["Wh"].shape[1:])
        y_train = np.asarray(handle["Wh"][np.sort(idx_train)], dtype=np.float32)
        schema_version = str(handle.attrs.get("schema_version", ""))
    if len(grid_shape) != 2 or len(target_shape) != 2:
        raise ValueError(f"unexpected grid/target shapes: {grid_shape}, {target_shape}")
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
        "idx_val": idx_val,
        "idx_test": idx_test,
        "target_mean": target_mean,
        "target_std": target_std,
        **hashes,
    }


def _evaluate(model: Any, loader: Any, *, device: Any, mean: Any, std: Any, torch: Any) -> dict[str, float]:
    model.eval()
    squared = 0.0
    absolute = 0.0
    relative = 0.0
    normalized_squared = 0.0
    count = 0
    inference_start = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            prediction_norm = model(xb)
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
            count += int(target.numel())
    elapsed = time.perf_counter() - inference_start
    band_count = count * max(int(mean.shape[-1]) - 1, 0) / max(int(mean.shape[-1]), 1)
    return {
        "mse": squared / max(count, 1),
        "normalized_mse": normalized_squared / max(count, 1),
        "mae": absolute / max(count, 1),
        "rmse": float(np.sqrt(squared / max(count, 1))),
        "bands_2_10_mre_pct": 100.0 * relative / max(band_count, 1),
        "eval_time_s": elapsed,
    }


def _plot_history(run_dir: Path, history: list[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train normalized MSE")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in history], label="validation normalized MSE")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [float(row["val_bands_2_10_mre_pct"]) for row in history])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation bands 2-10 MRE (%)")
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
) -> Path:
    """Run one isolated model-plus-seed job from a frozen wave config."""

    config_file = Path(config_path)
    resolved = _validate_and_resolve(
        _load_yaml(config_file),
        model_name=model_name,
        seed=seed,
        admission_probe=admission_probe,
        threshold_override=threshold,
        resolution_override=resolution,
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
    layout = str(representation["layout"])
    input_shape = contract["grid_shape"] if layout == "image" else (int(np.prod(contract["grid_shape"])),)
    model = build_model(
        model_name,
        input_shape=input_shape,
        output_shape=contract["target_shape"],
        **dict(resolved["model"]["params"]),
    ).to(device)
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
    train_dataset = _FullCellH5Dataset(indices=contract["idx_train"], **dataset_kwargs)
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
    accumulation = int(training.get("effective_batch_size", 64)) // int(training.get("micro_batch_size", 64))
    if accumulation < 1 or int(training.get("effective_batch_size", 64)) % int(training.get("micro_batch_size", 64)):
        raise ValueError("effective_batch_size must be an integer multiple of micro_batch_size")

    history: list[dict[str, Any]] = []
    best_mre = float("inf")
    best_epoch = 0
    stale_epochs = 0
    train_start = time.perf_counter()
    max_batches = training.get("max_train_batches")
    for epoch in range(1, int(training["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_count = 0
        for batch_index, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = loss_fn(model(xb), yb)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}, batch {batch_index}")
            (loss / accumulation).backward()
            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_sum += float(loss.detach().cpu()) * int(xb.shape[0])
            train_count += int(xb.shape[0])
            if max_batches is not None and batch_index >= int(max_batches):
                break
        val = _evaluate(model, val_loader, device=device, mean=mean_t, std=std_t, torch=torch)
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_loss": val["normalized_mse"],
            "val_bands_2_10_mre_pct": val["bands_2_10_mre_pct"],
            "learning_rate": float(training["learning_rate"]),
            "epoch_time_s": time.perf_counter() - epoch_start,
            "wall_time_s": time.perf_counter() - wall_start,
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6g val_mre=%.6g%% epoch_time_s=%.3f",
            epoch,
            row["train_loss"],
            row["val_bands_2_10_mre_pct"],
            row["epoch_time_s"],
        )
        if float(val["bands_2_10_mre_pct"]) < best_mre:
            best_mre = float(val["bands_2_10_mre_pct"])
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_tmp = out / "model.pt.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_params": dict(resolved["model"]["params"]),
                    "input_shape": input_shape,
                    "output_shape": contract["target_shape"],
                    "target_mean": torch.as_tensor(contract["target_mean"], dtype=torch.float32),
                    "target_std": torch.as_tensor(contract["target_std"], dtype=torch.float32),
                    "epoch": epoch,
                    "validation_bands_2_10_mre_pct": best_mre,
                    "resolved_config": resolved,
                },
                checkpoint_tmp,
            )
            os.replace(checkpoint_tmp, out / "model.pt")
        else:
            stale_epochs += 1
        _write_history(out / "history.csv", history)
        if not admission_probe and stale_epochs >= int(training.get("early_stopping_patience", 15)):
            logger.info("early stopping at epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    train_time = time.perf_counter() - train_start
    checkpoint = out / "model.pt"
    if not checkpoint.is_file():
        raise RuntimeError("best-validation checkpoint was not written")
    try:
        checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 2.1 supports the legacy default only on some builds.
        checkpoint_payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    final_val = _evaluate(model, val_loader, device=device, mean=mean_t, std=std_t, torch=torch)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device_info.is_cuda else 0.0
    metrics = {
        **final_val,
        "best_validation_bands_2_10_mre_pct": best_mre,
        "best_epoch": best_epoch,
        "stop_epoch": int(history[-1]["epoch"]),
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
    }
    _plot_history(out, history)
    write_metrics(out, metrics)
    status = "full_cell_forward_admission_probe_pass" if admission_probe else "formal_training_success"
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
            "selection_split": "validation",
            "test_executed": False,
            "model": model_name,
            "seed": int(seed),
            "lane": experiment.get("lane"),
            "representation": representation,
            "best_epoch": best_epoch,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "next_gate": "wave_admission_decision" if admission_probe else "12_job_wave_aggregation",
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
            "idx_val_hash": contract["idx_val_hash"],
            "idx_test_hash": contract["idx_test_hash"],
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
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["run_full_cell_forward"]
