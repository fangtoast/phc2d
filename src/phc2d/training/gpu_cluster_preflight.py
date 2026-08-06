"""Run an auditable single-GPU preflight on a full-cell HDF5 dataset.

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


class _H5SampleDataset:
    """Worker-safe lazy HDF5 reader used only for the throughput gate."""

    def __init__(self, path: Path, grid_field: str, indices: np.ndarray) -> None:
        self.path = Path(path)
        self.grid_field = str(grid_field)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._handle: Any | None = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> tuple[np.ndarray, np.ndarray]:
        if self._handle is None:
            import h5py

            self._handle = h5py.File(self.path, "r")
        index = int(self.indices[int(position)])
        return (
            np.asarray(self._handle[self.grid_field][index], dtype=np.float32),
            np.asarray(self._handle["Wh"][index], dtype=np.float32),
        )

    def __del__(self) -> None:  # pragma: no cover - worker lifecycle is process-owned
        if self._handle is not None:
            self._handle.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _read_hdf5_rows(
    path: Path,
    grid_field: str,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required for the full-cell GPU preflight") from exc

    selected = np.asarray(indices, dtype=np.int64)
    order = np.argsort(selected)
    sorted_indices = selected[order]
    with h5py.File(path, "r") as handle:
        if grid_field not in handle or "Wh" not in handle:
            raise KeyError(f"HDF5 requires '{grid_field}' and 'Wh': {path}")
        x_sorted = np.asarray(handle[grid_field][sorted_indices], dtype=np.float32)
        y_sorted = np.asarray(handle["Wh"][sorted_indices], dtype=np.float32)
        x = np.empty_like(x_sorted)
        y = np.empty_like(y_sorted)
        x[order] = x_sorted
        y[order] = y_sorted
        schema = {
            "schema_version": str(handle.attrs.get("schema_version", "")),
            "grid_shape": list(handle[grid_field].shape),
            "Wh_shape": list(handle["Wh"].shape),
            "idx_train_count": int(handle["idx_train"].shape[0]),
            "idx_val_count": int(handle["idx_val"].shape[0]),
            "idx_test_count": int(handle["idx_test"].shape[0]),
        }
    return x, y, schema


def _nvidia_driver() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None


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


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError("GPU preflight config must be a mapping")
    return dict(payload)


def run_gpu_cluster_preflight(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
) -> Path:
    """Execute the full single-card gate and write standard run artifacts."""

    config_file = Path(config_path)
    config = _load_config(config_file)
    experiment = dict(config.get("experiment") or {})
    dataset_cfg = dict(config.get("dataset") or {})
    model_cfg = dict(config.get("model") or {})
    training_cfg = dict(config.get("training") or {})
    runtime_cfg = dict(config.get("runtime") or {})
    output_cfg = dict(config.get("output") or {})

    dataset_path = Path(str(dataset_cfg["path"]))
    grid_field = str(dataset_cfg.get("grid_field", "material_weight_grid_64"))
    max_samples = int(dataset_cfg.get("preflight_samples", 256))
    throughput_samples = int(dataset_cfg.get("throughput_samples", 1024))
    batch_size = int(training_cfg.get("batch_size", 64))
    fno_epochs = int(training_cfg.get("fno_epochs", 10))
    num_workers = int(runtime_cfg.get("hdf5_num_workers", 4))
    require_cuda = bool(runtime_cfg.get("require_cuda", True))
    candidates = tuple(str(name) for name in model_cfg.get("candidates", ()))
    required_models = {"mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2"}
    if set(candidates) != required_models:
        raise ValueError(f"GPU preflight model set must be exactly {sorted(required_models)}")
    if batch_size != 64 and require_cuda:
        raise ValueError("Formal GPU preflight must test batch_size=64")
    if fno_epochs != 10 and require_cuda:
        raise ValueError("Formal GPU preflight must run exactly 10 FNO epochs")

    task = str(experiment.get("experiment_id", "full_cell_gpu_single_card_preflight"))
    out = make_run_dir(task, runs_dir=output_cfg.get("runs_dir", "runs"), run_dir=run_dir)
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "python -m phc2d.training.gpu_cluster_preflight",
            "config_path": str(config_file),
            "dataset_path": str(dataset_path),
            "validation_boundary": "single_gpu_environment_preflight_no_formal_training_no_mpb",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    write_resolved_config(out, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)

    wall_start = time.perf_counter()
    try:
        import h5py
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ImportError("torch and h5py are required for the GPU preflight") from exc

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal GPU preflight; CPU fallback is forbidden")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with h5py.File(dataset_path, "r") as handle:
        idx_train = np.asarray(handle["idx_train"], dtype=np.int64)
        idx_val = np.asarray(handle["idx_val"], dtype=np.int64)
        idx_test = np.asarray(handle["idx_test"], dtype=np.int64)
    if max_samples < batch_size:
        raise ValueError("preflight_samples must be at least batch_size")
    selected = idx_train[: min(max_samples, idx_train.size)]
    x, y, schema = _read_hdf5_rows(dataset_path, grid_field, selected)
    target_mean = np.mean(y, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    target_std = np.std(y, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    target_std = np.maximum(target_std, np.float32(1.0e-6))
    y_normalized = (y - target_mean) / target_std

    io_indices = idx_train[: min(throughput_samples, idx_train.size)]
    io_dataset = _H5SampleDataset(dataset_path, grid_field, io_indices)
    io_loader = DataLoader(
        io_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    io_start = time.perf_counter()
    io_count = 0
    for input_batch, _ in io_loader:
        io_count += int(input_batch.shape[0])
    io_elapsed = time.perf_counter() - io_start

    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_normalized, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    first_x, first_y = next(iter(train_loader))
    overrides = dict(model_cfg.get("overrides") or {})
    model_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    fno_model: Any | None = None
    loss_fn = torch.nn.MSELoss()
    for name in candidates:
        params = dict(overrides.get(name) or {})
        model = build_model(
            name,
            input_shape=tuple(int(v) for v in x.shape[1:]),
            output_shape=tuple(int(v) for v in y.shape[1:]),
            **params,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prediction = model(first_x.to(device, non_blocking=True))
        loss = loss_fn(prediction, first_y.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Non-finite forward/backward loss for {name}")
        peak_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        )
        model_rows.append(
            {
                "model": name,
                "status": "forward_backward_pass",
                "loss": float(loss.detach().cpu()),
                "output_shape": list(prediction.shape),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "peak_gpu_memory_mb": peak_mb,
            }
        )
        if name == "fno2d_band_v2":
            fno_model = model

    if fno_model is None:  # pragma: no cover - protected by exact model-set gate
        raise RuntimeError("fno2d_band_v2 was not constructed")
    fno_optimizer = torch.optim.AdamW(fno_model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()
    for epoch in range(1, fno_epochs + 1):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        seen = 0
        fno_model.train()
        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            fno_optimizer.zero_grad(set_to_none=True)
            prediction = fno_model(input_batch)
            loss = loss_fn(prediction, target_batch)
            loss.backward()
            fno_optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(input_batch.shape[0])
            seen += int(input_batch.shape[0])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "epoch_time_s": time.perf_counter() - epoch_start,
            }
        )
        logger.info("FNO preflight epoch %d/%d loss=%.6g", epoch, fno_epochs, history[-1]["train_loss"])
    train_time_s = time.perf_counter() - train_start
    peak_fno_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )

    checkpoint_path = out / "fno_preflight_checkpoint.pt"
    torch.save(fno_model.state_dict(), checkpoint_path)
    reloaded = build_model(
        "fno2d_band_v2",
        input_shape=tuple(int(v) for v in x.shape[1:]),
        output_shape=tuple(int(v) for v in y.shape[1:]),
        **dict(overrides.get("fno2d_band_v2") or {}),
    ).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    fno_model.eval()
    reloaded.eval()
    with torch.no_grad():
        probe = first_x.to(device, non_blocking=True)
        original_prediction = fno_model(probe).detach().cpu().numpy()
        reloaded_prediction = reloaded(probe).detach().cpu().numpy()
    checkpoint_match = bool(np.array_equal(original_prediction, reloaded_prediction))
    if not checkpoint_match:
        raise RuntimeError("Reloaded FNO checkpoint does not reproduce the saved model output")

    device_properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    metrics: dict[str, Any] = {
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "train_time_s": float(train_time_s),
        "hdf5_read_time_s": float(io_elapsed),
        "hdf5_throughput_samples_per_s": float(io_count / io_elapsed) if io_elapsed > 0 else 0.0,
        "hdf5_samples_read": int(io_count),
        "hdf5_num_workers": int(num_workers),
        "batch_size": int(batch_size),
        "fno_preflight_epochs": int(fno_epochs),
        "fno_peak_gpu_memory_mb": peak_fno_mb,
        "checkpoint_reload_exact_match": checkpoint_match,
        "requested_device": "cuda:0" if require_cuda else str(device),
        "resolved_device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "device_total_memory_mb": (
            float(device_properties.total_memory / (1024**2)) if device_properties is not None else 0.0
        ),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "nccl_version": list(torch.cuda.nccl.version()) if device.type == "cuda" else None,
        "nvidia_driver_version": _nvidia_driver(),
        "dataset_sha256": _sha256_file(dataset_path),
        "idx_train_hash": _sha256_array(idx_train),
        "idx_val_hash": _sha256_array(idx_val),
        "idx_test_hash": _sha256_array(idx_test),
        "dataset_schema": schema,
    }
    _write_history(out / "history.csv", history)
    write_metrics(out, metrics)
    write_results(
        out,
        {
            "status": "gpu_single_card_preflight_pass",
            "formal_training_started": False,
            "mpb_meep_used": False,
            "dataset_path": str(dataset_path),
            "grid_field": grid_field,
            "model_checks": model_rows,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "prediction_sha256": hashlib.sha256(original_prediction.tobytes()).hexdigest(),
            "next_gate": "three_seed_formal_training_authorization",
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.training.gpu_cluster_preflight",
            "config_path": str(config_file),
            "dataset_path": str(dataset_path),
            "validation_boundary": "single_gpu_environment_preflight_no_formal_training_no_mpb",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
            "wall_time_total_s": metrics["wall_time_total_s"],
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"GPU preflight artifact contract is incomplete: {missing}")
    logger.info("GPU single-card preflight passed: %s", out)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    print(run_gpu_cluster_preflight(args.config, run_dir=args.run_dir))


if __name__ == "__main__":
    main()


__all__ = ["run_gpu_cluster_preflight"]
