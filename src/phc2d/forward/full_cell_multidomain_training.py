"""Pooled multi-domain Full-Wh training for EXP-FWD-021.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phc2d.forward.full_cell_evaluation import evaluate_full_cell
from phc2d.forward.full_cell_multidomain_contracts import (
    read_csv,
    sha256_array,
    sha256_file,
    write_csv,
)
from phc2d.models import build_model, describe_model_identity, get_model_spec
from phc2d.runs import (
    ensure_figures_dir,
    setup_logging,
    write_metrics,
    write_resolved_config,
    write_results,
)
from phc2d.training.device import DeviceConfig, resolve_device


class MultiDomainH5Dataset:
    """Worker-safe read-only adapter for one frozen domain subset."""

    def __init__(
        self,
        path: str | Path,
        indices: np.ndarray,
        *,
        grid_field: str,
        layout: str,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.path = Path(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.grid_field = str(grid_field)
        self.layout = str(layout)
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
        if self.layout == "flat":
            x = x.reshape(-1, order="C")
        y = np.asarray(self._handle["Wh"][index], dtype=np.float32)
        return x, (y - self.target_mean) / self.target_std

    def __del__(self) -> None:  # pragma: no cover - worker lifecycle
        if self._handle is not None:
            self._handle.close()


def forward_full_wh(
    model: Any,
    inputs: Any,
    *,
    model_spec: Any,
    k_points: Any,
    lane_index: int | None,
    torch: Any,
) -> Any:
    """Forward an unconditioned Full-Wh model through its public registry contract."""

    if model_spec.accepts_lane_id or lane_index is not None:
        raise ValueError("EXP-FWD-021 forbids lane-conditioned models")
    kwargs = {"k_points": k_points} if model_spec.accepts_k_points else {}
    return model(inputs, **kwargs)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping in {path}")
    return dict(payload)


def _load_campaign(campaign_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    campaign = Path(campaign_dir).resolve()
    profile = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    if not isinstance(profile, Mapping):
        raise TypeError("resolved campaign config must be a mapping")
    return campaign, dict(profile)


def _task_row(campaign: Path, task_index: int) -> dict[str, str]:
    rows = read_csv(campaign / "task_manifest.csv")
    matches = [row for row in rows if int(row["task_index"]) == int(task_index)]
    if len(matches) != 1:
        raise RuntimeError(f"task_index {task_index} must resolve exactly once")
    return matches[0]


def _domain_rows(campaign: Path) -> dict[str, dict[str, str]]:
    return {row["domain"]: row for row in read_csv(campaign / "dataset_manifest.csv")}


def _normalizer(campaign: Path, regime: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    rows = [row for row in read_csv(campaign / "normalizer_manifest.csv") if row["regime"] == regime]
    if len(rows) != 1:
        raise RuntimeError(f"normalizer for {regime} must resolve exactly once")
    row = rows[0]
    source = campaign / row["normalizer_path"]
    if sha256_file(source) != row["normalizer_file_sha256"]:
        raise RuntimeError(f"normalizer file hash mismatch for {regime}")
    with np.load(source) as payload:
        mean = np.asarray(payload["target_mean"], dtype=np.float32)
        std = np.asarray(payload["target_std"], dtype=np.float32)
    if sha256_array(mean, dtype=np.float32) != row["target_mean_sha256"]:
        raise RuntimeError(f"normalizer mean hash mismatch for {regime}")
    if sha256_array(std, dtype=np.float32) != row["target_std_sha256"]:
        raise RuntimeError(f"normalizer std hash mismatch for {regime}")
    return mean, std, row


def _checkpoint_payload(
    *,
    model: Any,
    model_name: str,
    model_params: Mapping[str, Any],
    model_spec: Any,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    k_points: Any,
    k_points_sha256: str,
    mean: Any,
    std: Any,
    epoch: int,
    selection_mre: float,
    task: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "model_params": dict(model_params),
        "model_identity": describe_model_identity(
            model, model_name, model_spec, model_kwargs=model_params
        ),
        "input_shape": tuple(int(value) for value in input_shape),
        "output_shape": tuple(int(value) for value in output_shape),
        "k_points": k_points.detach().cpu() if k_points is not None else None,
        "k_points_sha256": k_points_sha256,
        "target_mean": mean.detach().cpu(),
        "target_std": std.detach().cpu(),
        "normalizer_kind": "pooled_selected_training_samples_per_coordinate",
        "epoch": int(epoch),
        "selection_split": "equal_source_domain_validation_macro",
        "validation_bands_2_10_mre_pct": float(selection_mre),
        "task": dict(task),
        "multidomain_provenance": dict(provenance),
    }


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _convergence(history: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    by_epoch = {int(row["epoch"]): float(row["val_bands_2_10_mre_pct"]) for row in history}
    required = tuple(range(181, 201))
    if not set(required).issubset(by_epoch):
        raise RuntimeError("terminal convergence history is incomplete")
    first = np.asarray([by_epoch[index] for index in range(181, 191)], dtype=np.float64)
    second = np.asarray([by_epoch[index] for index in range(191, 201)], dtype=np.float64)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise RuntimeError("terminal convergence history is non-finite")
    mean_first = float(first.mean())
    mean_second = float(second.mean())
    relative_change = abs(mean_second - mean_first) / max(abs(mean_first), 1.0e-12)
    return {
        "metric": "equal_source_domain_validation_bands_2_10_mre_pct",
        "first_window": [181, 190],
        "second_window": [191, 200],
        "mean_first_window": mean_first,
        "mean_second_window": mean_second,
        "relative_change": relative_change,
        "threshold": float(threshold),
        "passed": bool(relative_change <= threshold),
    }


def run_training_task(
    campaign_dir: str | Path,
    task_index: int,
    *,
    admission_probe: bool = False,
    device_override: str | None = None,
    max_train_batches: int | None = None,
    max_validation_samples: int | None = None,
) -> Path:
    """Run one manifest-addressed task from scratch without reading held-out test."""

    campaign, profile = _load_campaign(campaign_dir)
    task = _task_row(campaign, task_index)
    regime = task["regime"]
    model_name = task["model"]
    seed = int(task["seed"])
    output = (
        campaign / "admission" / f"{model_name}_s{seed}"
        if admission_probe
        else campaign / task["run_dir"]
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"training run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    ensure_figures_dir(output)
    logger = setup_logging(
        log_file=output / "run.log",
        run_meta={
            "script": "python -m phc2d.forward.full_cell_multidomain_campaign run-training-task",
            "experiment_id": "EXP-FWD-021",
            "task_id": task["task_id"],
            "admission_probe": bool(admission_probe),
            "execution_start_git_commit": task["execution_start_git_commit"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
        },
    )
    runtime = dict(profile["runtime"])
    if device_override is not None:
        runtime["device"] = device_override
        runtime["require_accelerator"] = str(device_override).startswith("cuda")
        if str(device_override) == "cpu":
            runtime["num_workers"] = 0
            runtime["pin_memory"] = False
    training = dict(profile["training"])
    training_epochs = 1 if admission_probe else int(training["max_epochs"])
    resolved = {
        **profile,
        "runtime": runtime,
        "active_task": dict(task),
        "admission_probe": bool(admission_probe),
        "training": {**training, "executed_epochs": training_epochs},
    }
    write_resolved_config(output, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))

    import h5py
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch)
    device = device_info.torch_device
    mean, std, normalizer_row = _normalizer(campaign, regime)
    datasets = _domain_rows(campaign)
    subset_rows = [
        row for row in read_csv(campaign / "subset_manifest.csv") if row["regime"] == regime
    ]
    if sum(int(row["sample_count"]) for row in subset_rows) != 8000:
        raise RuntimeError(f"{regime} frozen subset is not exactly 8,000 samples")
    model_spec = get_model_spec(model_name)
    layout = model_spec.input_layout
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    source_datasets = []
    validation_datasets: dict[str, MultiDomainH5Dataset] = {}
    source_counts: dict[str, int] = {}
    subset_hashes: dict[str, str] = {}
    dataset_hashes: dict[str, str] = {}
    k_points_np: np.ndarray | None = None
    input_grid_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    for subset in subset_rows:
        domain = subset["domain"]
        dataset = datasets[domain]
        indices = np.load(campaign / subset["indices_path"])
        if sha256_array(indices, dtype=np.int64) != subset["indices_sha256"]:
            raise RuntimeError(f"subset hash mismatch for {regime}/{domain}")
        path = Path(dataset["dataset_path"])
        with h5py.File(path, "r") as handle:
            idx_val = np.asarray(handle["idx_val"], dtype=np.int64)
            k_points = np.asarray(handle["k_points"], dtype=np.float32)
            grid_shape = tuple(int(value) for value in handle[dataset["grid_field"]].shape[1:])
            target_shape = tuple(int(value) for value in handle["Wh"].shape[1:])
        if np.intersect1d(indices, idx_val).size:
            raise RuntimeError(f"training-validation overlap for {regime}/{domain}")
        if k_points_np is not None and not np.array_equal(k_points_np, k_points):
            raise RuntimeError("source domains use different k-point grids")
        k_points_np = k_points
        input_grid_shape = grid_shape
        output_shape = target_shape
        kwargs = {
            "path": path,
            "grid_field": dataset["grid_field"],
            "layout": layout,
            "target_mean": mean,
            "target_std": std,
        }
        source_datasets.append(MultiDomainH5Dataset(indices=indices, **kwargs))
        validation_indices = idx_val[:max_validation_samples] if max_validation_samples else idx_val
        validation_datasets[domain] = MultiDomainH5Dataset(indices=validation_indices, **kwargs)
        source_counts[domain] = int(len(indices))
        subset_hashes[domain] = subset["indices_sha256"]
        dataset_hashes[domain] = dataset["dataset_sha256"]
    if input_grid_shape is None or output_shape is None or k_points_np is None:
        raise RuntimeError("training task has no source domains")
    input_shape = input_grid_shape if layout == "image" else (int(np.prod(input_grid_shape)),)
    model_params = dict(profile["models"][model_name])
    model_kwargs = {**model_params, "input_shape": input_shape, "output_shape": output_shape}
    if model_spec.accepts_k_points:
        model_kwargs["k_points"] = k_points_np
    model = build_model(model_name, **model_kwargs).to(device)
    model_identity = describe_model_identity(
        model, model_name, model_spec, model_kwargs=model_params
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(task["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_fn = torch.nn.MSELoss()
    loader_kwargs = {
        "batch_size": int(training["micro_batch_size"]),
        "num_workers": int(device_info.num_workers),
        "pin_memory": bool(device_info.pin_memory),
    }
    generator = torch.Generator().manual_seed(seed)
    train_dataset = ConcatDataset(source_datasets)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs)
    validation_loaders = {
        domain: DataLoader(dataset, shuffle=False, **loader_kwargs)
        for domain, dataset in validation_datasets.items()
    }
    k_points_t = torch.as_tensor(k_points_np, dtype=torch.float32, device=device)
    provenance = {
        "schema_version": "phc2d.full_cell_multidomain.v1",
        "experiment_id": "EXP-FWD-021",
        "regime": regime,
        "source_domains": list(validation_datasets),
        "source_sample_counts": source_counts,
        "subset_hashes": subset_hashes,
        "dataset_hashes": dataset_hashes,
        "normalizer_file_sha256": normalizer_row["normalizer_file_sha256"],
        "target_mean_sha256": normalizer_row["target_mean_sha256"],
        "target_std_sha256": normalizer_row["target_std_sha256"],
        "test_executed": False,
    }
    history: list[dict[str, Any]] = []
    best_mre = float("inf")
    best_epoch = 0
    optimizer_steps = 0
    samples_seen = 0
    wall_start = time.perf_counter()
    for epoch in range(1, training_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_index, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_full_wh(
                model,
                inputs,
                model_spec=model_spec,
                k_points=k_points_t,
                lane_index=None,
                torch=torch,
            )
            loss = loss_fn(prediction, targets)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}, batch {batch_index}")
            optimizer.step()
            optimizer_steps += 1
            batch_size = int(inputs.shape[0])
            samples_seen += batch_size
            train_count += batch_size
            train_sum += float(loss.detach().cpu()) * batch_size
            if max_train_batches is not None and batch_index >= int(max_train_batches):
                break
        domain_metrics: dict[str, dict[str, Any]] = {}
        for domain, loader in validation_loaders.items():
            domain_metrics[domain] = evaluate_full_cell(
                model,
                loader,
                device=device,
                mean=mean_t,
                std=std_t,
                model_spec=model_spec,
                k_points=k_points_t,
                lane_index=None,
                torch_module=torch,
                forward_model=forward_full_wh,
            )
        macro = {
            key: float(np.mean([values[key] for values in domain_metrics.values()]))
            for key in ("normalized_mse", "band1_mre_pct", "bands_2_10_mre_pct", "bands_1_10_mre_pct")
        }
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_loss": macro["normalized_mse"],
            "val_band1_mre_pct": macro["band1_mre_pct"],
            "val_bands_2_10_mre_pct": macro["bands_2_10_mre_pct"],
            "val_bands_1_10_mre_pct": macro["bands_1_10_mre_pct"],
            "learning_rate": float(task["learning_rate"]),
            "epoch_time_s": time.perf_counter() - epoch_start,
            "wall_time_s": time.perf_counter() - wall_start,
            "optimizer_steps": optimizer_steps,
            "samples_seen": samples_seen,
        }
        for domain, values in domain_metrics.items():
            row[f"{domain}_val_bands_2_10_mre_pct"] = values["bands_2_10_mre_pct"]
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6g source_macro_mre=%.6g%%",
            epoch,
            row["train_loss"],
            row["val_bands_2_10_mre_pct"],
        )
        if row["val_bands_2_10_mre_pct"] < best_mre:
            best_mre = float(row["val_bands_2_10_mre_pct"])
            best_epoch = epoch
            payload = _checkpoint_payload(
                model=model,
                model_name=model_name,
                model_params=model_params,
                model_spec=model_spec,
                input_shape=input_shape,
                output_shape=output_shape,
                k_points=k_points_t,
                k_points_sha256=next(iter(datasets.values()))["k_points_sha256"],
                mean=mean_t,
                std=std_t,
                epoch=epoch,
                selection_mre=best_mre,
                task=task,
                provenance=provenance,
            )
            _atomic_torch_save(torch, payload, output / "model.pt")
        write_csv(output / "history.csv", history)
    terminal = _checkpoint_payload(
        model=model,
        model_name=model_name,
        model_params=model_params,
        model_spec=model_spec,
        input_shape=input_shape,
        output_shape=output_shape,
        k_points=k_points_t,
        k_points_sha256=next(iter(datasets.values()))["k_points_sha256"],
        mean=mean_t,
        std=std_t,
        epoch=training_epochs,
        selection_mre=float(history[-1]["val_bands_2_10_mre_pct"]),
        task=task,
        provenance=provenance,
    )
    terminal["checkpoint_role"] = "admission_terminal" if admission_probe else "terminal_epoch"
    terminal_path = output / ("admission_epoch1.pt" if admission_probe else "epoch200.pt")
    _atomic_torch_save(torch, terminal, terminal_path)
    convergence = None if admission_probe else _convergence(
        history, float(training["convergence_relative_threshold"])
    )
    checkpoint = output / "model.pt"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_payload["model_identity"]["architecture_fingerprint"] != model_identity["architecture_fingerprint"]:
        raise RuntimeError("best checkpoint model identity mismatch")
    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device_info.is_cuda
        else 0.0
    )
    metrics = {
        "best_validation_bands_2_10_mre_pct": best_mre,
        "best_epoch": best_epoch,
        "stop_epoch": training_epochs,
        "history_epoch_count": len(history),
        "train_time_s": time.perf_counter() - wall_start,
        "peak_gpu_memory_mb": peak_memory,
        "parameter_count": model_identity["parameter_count"],
        "training_sample_count": len(train_dataset),
        "source_validation_sample_counts": {
            domain: len(dataset) for domain, dataset in validation_datasets.items()
        },
        "effective_batch_size": int(training["effective_batch_size"]),
        "micro_batch_size": int(training["micro_batch_size"]),
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "convergence": convergence,
        "device": device_info.metadata(peak_gpu_memory_mb=peak_memory),
        "model_identity": model_identity,
    }
    write_metrics(output, metrics)
    (output / "model_summary.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "model_identity": model_identity,
                "input_shape": list(input_shape),
                "output_shape": list(output_shape),
                "best_validation_checkpoint": str(checkpoint),
                "best_validation_checkpoint_sha256": sha256_file(checkpoint),
                "terminal_checkpoint": str(terminal_path),
                "terminal_checkpoint_sha256": sha256_file(terminal_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = "multidomain_admission_probe_pass"
    if not admission_probe:
        status = (
            "formal_training_success"
            if bool(convergence and convergence["passed"])
            else "training_complete_convergence_not_met"
        )
    write_results(
        output,
        {
            "status": status,
            "experiment_id": "EXP-FWD-021",
            "task_id": task["task_id"],
            "regime": regime,
            "model": model_name,
            "seed": seed,
            "from_scratch": True,
            "test_executed": False,
            "formal_training_started": not admission_probe,
            "checkpoint_selection": "equal_source_domain_validation_bands_2_10_mre",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "terminal_checkpoint_path": str(terminal_path),
            "terminal_checkpoint_sha256": sha256_file(terminal_path),
            "best_epoch": best_epoch,
            "stop_epoch": training_epochs,
            "convergence": convergence,
            "multidomain_provenance": provenance,
            "scientific_boundary": "admission_only" if admission_probe else "seed0_exploratory_surrogate_only",
            "model_promotion": False,
            "physical_claim": False,
        },
    )
    return output


__all__ = ["MultiDomainH5Dataset", "forward_full_wh", "run_training_task"]
