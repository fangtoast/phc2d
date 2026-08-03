"""B3 active-component supervised and adapted-PINO eigenfield workflows.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.models import architecture_fingerprint, build_model, describe_model_identity, get_model_spec
from phc2d.physics import canonicalize_complex_efield, derive_epsilon_on_field_grid, derive_epsilon_on_field_grid_torch, evaluate_te_bloch_eigen_residual, evaluate_te_bloch_eigen_residual_torch, pino_te_physics_loss
from phc2d.runs import ensure_figures_dir, make_run_dir, setup_logging, write_json, write_metrics, write_resolved_config, write_results, write_run_meta
from phc2d.training import DeviceConfig, collect_device_metadata, peak_gpu_memory_mb, resolve_device, synchronize_device


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FieldEigenData:
    grid: np.ndarray
    context: np.ndarray
    target: np.ndarray
    frequency: np.ndarray
    epsilon: np.ndarray
    k_point: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    record_id: np.ndarray
    source_id: np.ndarray
    split: np.ndarray
    indices: dict[str, np.ndarray]
    dataset_hash: str
    field_shape: tuple[int, int]


@dataclass(frozen=True)
class DenseFrequencyData:
    """Train-only dense-k frequency contexts bound to B3 source structures."""

    source_grid: np.ndarray
    source_epsilon: np.ndarray
    source_index: np.ndarray
    context: np.ndarray
    frequency: np.ndarray
    source_ids: np.ndarray
    v2_dataset_hash: str
    k_path_hash: str


def is_field_eigen_v2_config(path: str | Path) -> bool:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return str(dict(raw.get("experiment") or {}).get("execution", "")).lower() == "field_eigen_v2"


def _torch():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    return torch, DataLoader, TensorDataset


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])


def complex_to_active_channels(values: np.ndarray) -> np.ndarray:
    """Encode complex ``Ex/Ey/Hz`` as real channels without inactive TE zeros."""

    array = np.asarray(values)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError("Active TE values must have shape (sample, nx, ny, 3).")
    return np.concatenate((array.real, array.imag), axis=-1).transpose(0, 3, 1, 2).astype(np.float32)


def active_channels_to_complex(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 4 or array.shape[1] != 6:
        raise ValueError("Active TE channels must have shape (sample, 6, nx, ny).")
    moved = array.transpose(0, 2, 3, 1)
    return moved[..., :3] + 1j * moved[..., 3:]


def reconstruct_full_te_fields(active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(active)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("Active TE complex fields must contain Ex/Ey/Hz.")
    shape = (*values.shape[:3], 1, 3)
    efield = np.zeros(shape, dtype=values.dtype)
    hfield = np.zeros(shape, dtype=values.dtype)
    efield[..., 0, :2] = values[..., :2]
    hfield[..., 0, 2] = values[..., 2]
    return efield, hfield


def _periodic_grid(epsilon: np.ndarray, nx: int, ny: int, *, include_coordinates: bool, epsilon_range: tuple[float, float]) -> np.ndarray:
    if nx % epsilon.shape[1] or ny % epsilon.shape[2]:
        raise ValueError("epsilon-to-field grid ratio is not integral.")
    low, high = epsilon_range
    lifted = np.repeat(np.repeat((epsilon - low) / max(high - low, 1.0e-12), nx // epsilon.shape[1], axis=1), ny // epsilon.shape[2], axis=2)
    channels = [lifted[:, None].astype(np.float32)]
    if include_coordinates:
        x = np.arange(nx, dtype=np.float32) / float(nx)
        y = np.arange(ny, dtype=np.float32) / float(ny)
        coordinate = np.stack(np.meshgrid(x, y, indexing="ij"), axis=0)
        periodic = np.stack((np.sin(2 * np.pi * coordinate[0]), np.cos(2 * np.pi * coordinate[0]), np.sin(2 * np.pi * coordinate[1]), np.cos(2 * np.pi * coordinate[1])), axis=0)
        channels.append(np.broadcast_to(periodic[None], (len(epsilon), 4, nx, ny)).astype(np.float32))
    return np.concatenate(channels, axis=1)


def load_field_eigen_v2(path: Path, cfg: Mapping[str, Any]) -> FieldEigenData:
    import h5py

    with h5py.File(path, "r") as handle:
        efield = np.asarray(handle["efield_complex"])
        hfield = np.asarray(handle["hfield_complex"])
        epsilon = np.asarray(handle["epsilon"], dtype=np.float32)
        k_point = np.asarray(handle["k_points"], dtype=np.float32)
        band = np.asarray(handle["band_index"], dtype=np.float32)
        frequency = np.asarray(handle["eigenfrequency"], dtype=np.float32)
        dx = np.asarray(handle["dx"], dtype=np.float32)
        dy = np.asarray(handle["dy"], dtype=np.float32)
        split = _decode(np.asarray(handle["source_split"]))
        record_id = _decode(np.asarray(handle["record_id"]))
        source_id = _decode(np.asarray(handle["source_sample_id"]))
        polarization = _decode(np.asarray(handle["polarization"]))
    if set(value.upper() for value in polarization) != {"TE"}:
        raise ValueError("B3 active component codec currently accepts only TE records.")
    if efield.shape != hfield.shape or efield.ndim != 5 or efield.shape[-2:] != (1, 3):
        raise ValueError("Field HDF5 must expose matching MPB Cartesian E/H tensors.")
    active = np.concatenate((efield[..., 0, :2], hfield[..., 0, 2:3]), axis=-1)
    target = complex_to_active_channels(active)
    nx, ny = int(efield.shape[1]), int(efield.shape[2])
    representation = dict(cfg.get("representation") or {})
    normalization = dict(cfg.get("normalization") or {})
    eps_range = tuple(float(value) for value in normalization.get("epsilon_range", (1.0, 8.9)))
    grid = _periodic_grid(epsilon, nx, ny, include_coordinates=bool(representation.get("periodic_coordinates", True)), epsilon_range=eps_range)
    k_scale = float(normalization.get("k_scale", 0.5))
    band_range = tuple(float(value) for value in normalization.get("band_range_one_based", (float(band.min()), float(band.max()))))
    context = np.stack((k_point[:, 0] / k_scale, k_point[:, 1] / k_scale, (band - band_range[0]) / max(band_range[1] - band_range[0], 1.0)), axis=1).astype(np.float32)
    indices = {name: np.flatnonzero(split == name).astype(np.int64) for name in ("train", "val", "test")}
    expected = dict(dict(cfg.get("dataset") or {}).get("expected_split_records") or {})
    if expected and {name: len(indices[name]) for name in indices} != {name: int(expected[name]) for name in indices}:
        raise ValueError("Field HDF5 does not match configured frozen source_split record counts.")
    source_sets = {name: set(source_id[index].tolist()) for name, index in indices.items()}
    if source_sets["train"] & source_sets["val"] or source_sets["train"] & source_sets["test"] or source_sets["val"] & source_sets["test"]:
        raise ValueError("source_sample_id leakage across field source_split.")
    return FieldEigenData(grid, context, target, frequency, epsilon, k_point, dx, dy, record_id, source_id, split, indices, _hash(path), (nx, ny))


def _source_index(source_id: str) -> int:
    try:
        return int(str(source_id).rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot map B3 source_sample_id {source_id!r} to V2 source index.") from exc


def _load_dense_frequency_supervision(
    data: FieldEigenData,
    cfg: Mapping[str, Any],
) -> DenseFrequencyData | None:
    """Load train-only V2 dense-k labels without adding frequency to model inputs."""

    training = dict(cfg.get("training") or {})
    raw = dict(training.get("dense_frequency_supervision") or {})
    if not bool(raw.get("enabled", False)):
        return None
    dataset_path = ROOT / str(raw["v2_frequency_dataset_path"])
    k_path_path = ROOT / str(raw["k_path_sidecar_path"])
    expected_dataset_hash = str(raw["v2_frequency_dataset_sha256"])
    expected_k_path_hash = str(raw["k_path_sidecar_sha256"])
    if _hash(dataset_path) != expected_dataset_hash:
        raise ValueError("Dense-frequency V2 dataset SHA256 mismatch.")
    if _hash(k_path_path) != expected_k_path_hash:
        raise ValueError("Dense-frequency k-path sidecar SHA256 mismatch.")
    sidecar = json.loads(k_path_path.read_text(encoding="utf-8"))
    points = np.asarray(sidecar.get("k_points"), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != int(sidecar.get("Nk", -1)):
        raise ValueError("Dense-frequency sidecar has invalid k_points.")
    with np.load(dataset_path, allow_pickle=False) as bundle:
        wh = np.asarray(bundle["Wh"], dtype=np.float32)
        idx_train = set(int(value) for value in np.asarray(bundle["idx_train"], dtype=np.int64).tolist())
    band_range = tuple(int(value) for value in dict(cfg.get("normalization") or {})["band_range_one_based"])
    bands = np.arange(band_range[0], band_range[1] + 1, dtype=np.int64)
    if wh.ndim != 3 or wh.shape[1] != len(points) or int(bands[-1]) > wh.shape[2]:
        raise ValueError("Dense-frequency Wh shape does not match configured k-path or band range.")
    train_sources = sorted(set(str(data.source_id[index]) for index in data.indices["train"]))
    source_to_record: dict[str, int] = {}
    for index in data.indices["train"]:
        source_to_record.setdefault(str(data.source_id[index]), int(index))
    if set(source_to_record) != set(train_sources):
        raise ValueError("Dense-frequency source map must contain exactly B3 train sources.")
    source_rows: list[int] = []
    source_ids: list[str] = []
    for source_id in train_sources:
        index = _source_index(source_id)
        if index not in idx_train:
            raise ValueError(f"Dense-frequency source {source_id!r} is not in the V2 train split.")
        source_rows.append(source_to_record[source_id]); source_ids.append(source_id)
    source_grid = data.grid[np.asarray(source_rows, dtype=np.int64)]
    source_epsilon = data.epsilon[np.asarray(source_rows, dtype=np.int64)]
    source_indices = np.repeat(np.arange(len(source_rows), dtype=np.int64), len(points) * len(bands))
    repeated_points = np.tile(np.repeat(points, len(bands), axis=0), (len(source_rows), 1))
    repeated_bands = np.tile(np.tile(bands, len(points)), len(source_rows))
    k_scale = float(dict(cfg.get("normalization") or {})["k_scale"])
    context = np.stack((repeated_points[:, 0] / k_scale, repeated_points[:, 1] / k_scale, (repeated_bands - bands[0]) / max(float(bands[-1] - bands[0]), 1.0)), axis=1).astype(np.float32)
    frequency = wh[np.repeat(np.asarray([_source_index(value) for value in source_ids], dtype=np.int64), len(points) * len(bands)), np.tile(np.repeat(np.arange(len(points)), len(bands)), len(source_rows)), np.tile(np.tile(bands - 1, len(points)), len(source_rows))]
    if not np.isfinite(frequency).all() or np.any(frequency <= 0.0):
        raise ValueError("Dense-frequency supervision contains non-positive or non-finite frequencies.")
    return DenseFrequencyData(source_grid, source_epsilon, source_indices, context, frequency.astype(np.float32), np.asarray(source_ids), expected_dataset_hash, expected_k_path_hash)


def _torch_active_complex(torch: Any, channels: Any) -> Any:
    moved = channels.permute(0, 2, 3, 1)
    return torch.complex(moved[..., :3], moved[..., 3:])


def phase_invariant_field_loss(
    torch: Any,
    prediction: Any,
    target: Any,
    *,
    epsilon: Any,
    e_weight: float = 0.5,
    h_weight: float = 0.5,
) -> tuple[Any, Any, Any, Any]:
    """Return a shared-phase field loss using the electric-energy metric."""

    pred = _torch_active_complex(torch, prediction)
    truth = _torch_active_complex(torch, target)
    eps = derive_epsilon_on_field_grid_torch(
        epsilon.to(dtype=pred.real.dtype, device=pred.device),
        (int(pred.shape[1]), int(pred.shape[2])),
    )
    inner = (pred[..., :2].conj() * eps[..., None] * truth[..., :2]).flatten(1).sum(1)
    phase = inner / inner.abs().clamp_min(1.0e-12)
    aligned = pred * phase[:, None, None, None]
    e_loss = (
        (eps[..., None] * (aligned[..., :2] - truth[..., :2]).abs().square()).flatten(1).sum(1)
        / (eps[..., None] * truth[..., :2].abs().square()).flatten(1).sum(1).clamp_min(1.0e-12)
    ).mean()
    h_loss = ((aligned[..., 2:] - truth[..., 2:]).abs().square().flatten(1).sum(1) / truth[..., 2:].abs().square().flatten(1).sum(1).clamp_min(1.0e-12)).mean()
    if e_weight < 0.0 or h_weight < 0.0 or e_weight + h_weight <= 0.0:
        raise ValueError("Field loss weights must be non-negative with a positive sum.")
    normalizer = float(e_weight + h_weight)
    return (float(e_weight) * e_loss + float(h_weight) * h_loss) / normalizer, e_loss, h_loss, aligned


def _data_loss(
    torch: Any,
    prediction: Any,
    frequency_z: Any,
    target: Any,
    target_frequency_z: Any,
    *,
    epsilon: Any,
    loss_weights: Mapping[str, float] | None = None,
):
    weights = dict(loss_weights or {})
    field_weight = float(weights.get("field", 1.0))
    frequency_weight = float(weights.get("frequency", 0.1))
    if field_weight < 0.0 or frequency_weight < 0.0:
        raise ValueError("Field and frequency loss weights must be non-negative.")
    field_loss, e_loss, h_loss, _ = phase_invariant_field_loss(
        torch,
        prediction,
        target,
        epsilon=epsilon,
        e_weight=float(weights.get("e", 0.5)),
        h_weight=float(weights.get("h", 0.5)),
    )
    frequency_loss = (frequency_z - target_frequency_z).square().mean()
    return field_weight * field_loss + frequency_weight * frequency_loss, e_loss, h_loss, frequency_loss


def _parameter_groups_for_gradient_audit(model: Any) -> dict[str, list[Any]]:
    """Return explicitly named frequency-head and field-decoder parameters."""

    named = list(model.named_parameters())
    frequency = [parameter for name, parameter in named if "frequency_head" in name]
    field = [parameter for name, parameter in named if "field_head" in name or "decoder" in name]
    if not frequency or not field:
        raise ValueError("B4 gradient audit requires identifiable frequency_head and field_head/decoder parameters.")
    return {"frequency_head": frequency, "field_decoder": field}


def _gradient_norm(torch: Any, gradients: list[Any | None]) -> float:
    squared = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.stack(squared).sum()).detach().cpu()) if squared else 0.0


def _pde_gradient_audit(torch: Any, pde_loss: Any, groups: Mapping[str, list[Any]]) -> dict[str, float]:
    """Measure PDE-only gradients without mutating parameter ``.grad`` buffers."""

    values: dict[str, float] = {}
    for name, parameters in groups.items():
        gradients = torch.autograd.grad(pde_loss, parameters, retain_graph=True, allow_unused=True)
        values[f"pde_{name}_grad_norm"] = _gradient_norm(torch, list(gradients))
    return values


def physics_frequency_from_prediction(predicted_frequency_z: Any, *, mode: str) -> Any:
    """Keep residual values predictive while optionally isolating its frequency gradient."""

    if mode == "full":
        return predicted_frequency_z
    if mode == "detach":
        return predicted_frequency_z.detach()
    raise ValueError("physics frequency gradient mode must be 'full' or 'detach'.")


def physics_frequency_for_residual(
    predicted_frequency_z: Any,
    target_frequency_z: Any,
    *,
    source: str,
    gradient_mode: str,
) -> Any:
    """Select the normalized frequency used by the PINO residual."""

    if source == "predicted":
        return physics_frequency_from_prediction(predicted_frequency_z, mode=gradient_mode)
    if source == "target":
        return target_frequency_z.detach()
    raise ValueError("physics frequency source must be 'predicted' or 'target'.")


def _periodic_grid_torch(torch: Any, epsilon: Any, *, resolution: int, epsilon_range: tuple[float, float]) -> Any:
    """Build the schema-equivalent periodic input at one requested PDE grid."""

    if epsilon.ndim != 3:
        raise ValueError("Multi-resolution PINO epsilon must have shape (batch, nx, ny).")
    nx, ny = int(epsilon.shape[1]), int(epsilon.shape[2])
    if int(resolution) % nx or int(resolution) % ny:
        raise ValueError("Configured PDE resolution must be an integral multiple of the material grid.")
    low, high = epsilon_range
    lifted = ((epsilon - low) / max(high - low, 1.0e-12)).repeat_interleave(int(resolution) // nx, dim=1).repeat_interleave(int(resolution) // ny, dim=2)
    coordinate = torch.arange(int(resolution), device=epsilon.device, dtype=epsilon.dtype) / float(resolution)
    x, y = torch.meshgrid(coordinate, coordinate, indexing="ij")
    periodic = torch.stack((torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * y), torch.cos(2 * torch.pi * y)), dim=0)
    return torch.cat((lifted[:, None], periodic[None].expand(int(epsilon.shape[0]), -1, -1, -1)), dim=1)


def _rayleigh_consistency_loss(torch: Any, *, physical_frequency: Any, residual: Mapping[str, Any]) -> Any:
    """Tie a frequency-head prediction to the eigenfrequency implied by its E field."""

    rayleigh = residual["rayleigh_frequency"]
    valid = residual["rayleigh_valid"] & torch.isfinite(rayleigh) & torch.isfinite(physical_frequency)
    if not bool(valid.any()):
        raise ValueError("Rayleigh consistency received no positive finite eigenfrequency estimates.")
    return (torch.log(physical_frequency[valid].clamp_min(1.0e-12)) - torch.log(rayleigh[valid].clamp_min(1.0e-12))).square().mean()


def _checkpoint_payload(
    *,
    model: Any,
    model_cfg: Mapping[str, Any],
    frequency_mean: float,
    frequency_std: float,
    dataset_hash: str,
    seed: int,
    regime: str,
    lambda_pde: float,
    epoch: int | None,
    validation_loss: float | None,
    deterministic: bool,
    loss_weights: Mapping[str, float],
    physics_frequency_gradient: str,
    physics_frequency_source: str,
    training_extensions: Mapping[str, Any] | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    payload = {
        "state_dict": model.state_dict(), "model_name": model_cfg["name"], "model_params": dict(model_cfg.get("params") or {}),
        "log_frequency_mean": frequency_mean, "log_frequency_std": frequency_std, "dataset_hash": dataset_hash,
        "seed": seed, "regime": regime, "lambda_pde": lambda_pde, "epoch": epoch,
        "validation_loss": validation_loss, "deterministic": deterministic, "loss_weights": dict(loss_weights),
        "physics_frequency_gradient": physics_frequency_gradient,
        "physics_frequency_source": physics_frequency_source,
        "training_extensions": dict(training_extensions or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def _lambda_at_epoch(epoch: int, target: float, warmup: int, ramp: int) -> float:
    if epoch <= warmup:
        return 0.0
    return float(target) * min(1.0, max(0.0, (epoch - warmup) / max(ramp, 1)))


def _model_forward(model: Any, model_name: str, grid: Any, context: Any):
    return model(grid, context) if str(model_name).endswith("_v2") else model(grid)


def _learning_rate_scale(epoch: int, epochs: int, warmup: int, minimum_ratio: float) -> float:
    if epoch < warmup:
        return max(1.0e-6, (epoch + 1) / max(warmup, 1))
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _epsilon_on_field_grid_numpy(epsilon: np.ndarray, field_shape: tuple[int, int]) -> np.ndarray:
    return np.stack([derive_epsilon_on_field_grid(item, field_shape) for item in np.asarray(epsilon)], axis=0)


def _phase_align_numpy(prediction: np.ndarray, target: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """Align a batch with the epsilon-weighted electric-field inner product."""

    eps = _epsilon_on_field_grid_numpy(epsilon, (int(prediction.shape[1]), int(prediction.shape[2])))
    inner = np.sum(np.conj(prediction[..., :2]) * eps[..., None] * target[..., :2], axis=(1, 2, 3))
    phase = inner / np.maximum(np.abs(inner), 1.0e-12)
    return prediction * phase[:, None, None, None]


def _relative_l2_numpy(truth: np.ndarray, prediction: np.ndarray, metric: np.ndarray | None = None) -> np.ndarray:
    weight = 1.0 if metric is None else np.asarray(metric)[..., None]
    numerator = np.sqrt(np.sum(weight * np.abs(prediction - truth) ** 2, axis=(1, 2, 3)))
    denominator = np.sqrt(np.sum(weight * np.abs(truth) ** 2, axis=(1, 2, 3)))
    # Correctly epsilon-normalized E labels have denominator*sqrt(dx*dy)=1.
    # Retain the measured denominator for legacy datasets and float32 drift.
    return numerator / np.maximum(denominator, 1.0e-12)


def _evaluate_metrics(data: FieldEigenData, prediction_channels: np.ndarray, frequency: np.ndarray, indices: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    target = active_channels_to_complex(data.target[indices])
    epsilon = data.epsilon[indices]
    epsilon_field = _epsilon_on_field_grid_numpy(epsilon, (int(target.shape[1]), int(target.shape[2])))
    prediction = _phase_align_numpy(active_channels_to_complex(prediction_channels), target, epsilon)
    e_relative = _relative_l2_numpy(target[..., :2], prediction[..., :2], epsilon_field)
    h_relative = _relative_l2_numpy(target[..., 2:], prediction[..., 2:])
    amplitude = np.mean(np.abs(np.abs(prediction) - np.abs(target)), axis=(1, 2, 3))
    mask = np.abs(target) >= 0.05 * np.max(np.abs(target), axis=(1, 2, 3), keepdims=True)
    phase_error = np.angle(np.exp(1j * (np.angle(prediction) - np.angle(target))))
    phase_mae = np.asarray([np.mean(np.abs(value[item_mask])) if np.any(item_mask) else np.nan for value, item_mask in zip(phase_error, mask)])
    frequency_mre = 100.0 * np.abs(frequency - data.frequency[indices]) / np.maximum(np.abs(data.frequency[indices]), 1.0e-12)
    residual_values, boundary_values, rows = [], [], []
    predicted_e, _ = reconstruct_full_te_fields(prediction)
    for position, original in enumerate(indices):
        try:
            canonical = canonicalize_complex_efield(predicted_e[position], epsilon=data.epsilon[original], dx=float(data.dx[original]), dy=float(data.dy[original]))
            residual = evaluate_te_bloch_eigen_residual(canonical, epsilon=data.epsilon[original], eigenfrequency=float(frequency[position]), dx=float(data.dx[original]), dy=float(data.dy[original]), bloch_k_point=data.k_point[original])
            residual_value, boundary_value = float(residual["relative_residual"]), float(residual["boundary_relative_residual"])
        except ValueError:
            residual_value = boundary_value = float("nan")
        residual_values.append(residual_value); boundary_values.append(boundary_value)
        rows.append({"record_id": str(data.record_id[original]), "source_sample_id": str(data.source_id[original]), "e_relative_l2": float(e_relative[position]), "h_relative_l2": float(h_relative[position]), "amplitude_mae": float(amplitude[position]), "phase_mae_rad": float(phase_mae[position]), "frequency_mre_pct": float(frequency_mre[position]), "maxwell_relative_residual": residual_value, "boundary_relative_residual": boundary_value})
    metrics = {"e_relative_l2": float(np.mean(e_relative)), "h_relative_l2": float(np.mean(h_relative)), "amplitude_mae": float(np.mean(amplitude)), "phase_mae_rad": float(np.nanmean(phase_mae)), "frequency_mre_pct": float(np.mean(frequency_mre)), "maxwell_relative_residual": float(np.nanmedian(residual_values)), "boundary_relative_residual": float(np.nanmedian(boundary_values)), "invalid_predicted_field_count": int(np.sum(~np.isfinite(residual_values)))}
    return metrics, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: int(value) if key == "epoch" else float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _latest_epoch_checkpoint(path: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    for checkpoint in path.glob("checkpoint_epoch_*.pt"):
        try:
            candidates.append((int(checkpoint.stem.rsplit("_", 1)[1]), checkpoint))
        except ValueError:
            continue
    return max(candidates, default=None, key=lambda item: item[0])


def _history_best_and_stale(history: list[dict[str, Any]]) -> tuple[float, int]:
    if not history:
        return float("inf"), 0
    best_index = min(range(len(history)), key=lambda index: float(history[index]["val_data_loss"]))
    return float(history[best_index]["val_data_loss"]), len(history) - best_index - 1


def _read(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw["_path"] = config_path
    return raw


def _validate_p6_protocol(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed when a P6 target-frequency ablation loses its frozen lineage."""

    protocol = dict(raw.get("p6_protocol") or {})
    if not protocol:
        return {}
    if str(protocol.get("only_named_change")) != "residual_frequency_source_predicted_to_target":
        raise ValueError("P6 protocol must freeze the target-frequency residual as its only named change.")
    if bool(protocol.get("inverse_authorized", False)) or bool(protocol.get("mpb_replay_authorized", False)):
        raise ValueError("P6 training cannot authorize inverse or MPB replay.")
    if str(dict(raw.get("evaluation") or {}).get("split")) != "val":
        raise ValueError("P6 target-frequency ablation must remain validation-only.")
    if str(dict(raw.get("training") or {}).get("physics_frequency_source")) != "target":
        raise ValueError("P6 target-frequency ablation must set physics_frequency_source=target.")
    lineage: dict[str, Any] = {}
    for name, path_key, hash_key in (
        ("source_config", "source_config", "source_config_sha256"),
        ("trigger_decision", "trigger_decision", "trigger_decision_sha256"),
    ):
        path = ROOT / str(protocol[path_key])
        expected = str(protocol[hash_key]).lower()
        actual = _hash(path)
        if actual != expected:
            raise ValueError(f"P6 {name} SHA256 mismatch.")
        lineage[name] = {"path": str(path), "sha256": actual}
    return {
        **lineage,
        "only_named_change": str(protocol["only_named_change"]),
        "inverse_authorized": False,
        "mpb_replay_authorized": False,
    }


def _run(raw: dict[str, Any], *, run_dir: Path | None = None, train: bool = True, checkpoint: Path | None = None) -> Path:
    torch, DataLoader, TensorDataset = _torch()
    p6_lineage = _validate_p6_protocol(raw)
    dataset_path = ROOT / str(dict(raw["dataset"])["path"])
    data = load_field_eigen_v2(dataset_path, raw)
    training = dict(raw["training"]); model_cfg = dict(raw["model"]); runtime = dict(raw["runtime"])
    seed = int(training.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(runtime.get("deterministic", True))
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch); device = device_info.torch_device
    train_index = data.indices["train"]
    dense_frequency = _load_dense_frequency_supervision(data, raw)
    normalization_frequency = data.frequency[train_index] if dense_frequency is None else np.concatenate((data.frequency[train_index], dense_frequency.frequency))
    frequency_mean = float(np.log(normalization_frequency.astype(np.float64)).mean()); frequency_std = max(float(np.log(normalization_frequency.astype(np.float64)).std()), 1.0e-8)
    log_frequency = np.log(data.frequency.astype(np.float64))
    frequency_z = ((log_frequency - frequency_mean) / frequency_std).astype(np.float32)
    experiment_id = str(dict(raw["experiment"])["experiment_id"])
    out = make_run_dir(experiment_id, runs_dir=ROOT / str(dict(raw.get("output") or {}).get("runs_dir", "runs")), run_dir=run_dir)
    logger = setup_logging(log_file=out / "run.log", run_meta={"stage": "field_eigen_v2"}); ensure_figures_dir(out)
    started = time.perf_counter()
    model = build_model(str(model_cfg["name"]), **dict(model_cfg.get("params") or {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", 3.0e-4)), weight_decay=float(training.get("weight_decay", 1.0e-5)))
    epochs = int(training.get("epochs", 200)); warmup_lr = int(training.get("lr_warmup_epochs", 5)); minimum_lr = float(training.get("minimum_learning_rate", 1.0e-6)); base_lr = float(training.get("learning_rate", 3.0e-4))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: _learning_rate_scale(epoch, epochs, warmup_lr, minimum_lr / base_lr))
    resumed_from_checkpoint = False; resumed_checkpoint_path: Path | None = None; resumed_checkpoint_epoch: int | None = None; optimizer_state_restored = False
    resume_payload: dict[str, Any] | None = None
    if train and checkpoint is None and not (out / "model.pt").is_file():
        latest_checkpoint = _latest_epoch_checkpoint(out)
        if latest_checkpoint is not None:
            resumed_checkpoint_epoch, resumed_checkpoint_path = latest_checkpoint
            resume_payload = torch.load(resumed_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_payload["state_dict"])
            frequency_mean = float(resume_payload["log_frequency_mean"]); frequency_std = float(resume_payload["log_frequency_std"])
            resumed_from_checkpoint = True
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["state_dict"]); frequency_mean = float(payload["log_frequency_mean"]); frequency_std = float(payload["log_frequency_std"])
    tensors = [torch.as_tensor(value) for value in (data.grid, data.context, data.target, frequency_z, data.epsilon, data.k_point, data.dx, data.dy)]
    regime = str(training.get("regime", "data_only")); lambda_target = float(training.get("lambda_pde", 0.0)); physics_warmup = int(training.get("physics_warmup_epochs", 20)); physics_ramp = int(training.get("physics_ramp_epochs", 20))
    loss_weights = {"field": 1.0, "frequency": 0.1, "e": 0.5, "h": 0.5, **dict(training.get("loss_weights") or {})}
    pde_resolutions = tuple(int(value) for value in training.get("physics_resolutions", [data.field_shape[0]]))
    if not pde_resolutions or len(set(pde_resolutions)) != len(pde_resolutions) or data.field_shape[0] not in pde_resolutions:
        raise ValueError("training.physics_resolutions must contain the unique supervised field resolution.")
    if any(value <= 1 for value in pde_resolutions):
        raise ValueError("training.physics_resolutions must contain positive spatial resolutions.")
    raw_reference_factors = dict(training.get("physics_solver_floor_factors") or {})
    solver_floor_factors = {resolution: float(raw_reference_factors.get(str(resolution), raw_reference_factors.get(resolution, 1.0))) for resolution in pde_resolutions}
    if any(value <= 0.0 or not math.isfinite(value) for value in solver_floor_factors.values()):
        raise ValueError("training.physics_solver_floor_factors must be finite and positive.")
    rayleigh_weight = float(training.get("rayleigh_consistency_weight", 0.0))
    if rayleigh_weight < 0.0:
        raise ValueError("training.rayleigh_consistency_weight must be non-negative.")
    extensions = {
        "dense_frequency_supervision": {
            "enabled": dense_frequency is not None,
            "v2_dataset_sha256": dense_frequency.v2_dataset_hash if dense_frequency else None,
            "k_path_sidecar_sha256": dense_frequency.k_path_hash if dense_frequency else None,
            "train_source_count": int(len(dense_frequency.source_ids)) if dense_frequency else 0,
            "context_count": int(len(dense_frequency.frequency)) if dense_frequency else 0,
        },
        "physics_resolutions": list(pde_resolutions),
        "physics_solver_floor_factors": {str(key): value for key, value in solver_floor_factors.items()},
        "rayleigh_consistency_weight": rayleigh_weight,
        "p6_protocol": p6_lineage,
    }
    frequency_gradient_mode = str(training.get("physics_frequency_gradient", "full")).lower()
    if frequency_gradient_mode not in {"full", "detach"}:
        raise ValueError("training.physics_frequency_gradient must be 'full' or 'detach'.")
    frequency_source = str(training.get("physics_frequency_source", "predicted")).lower()
    if frequency_source not in {"predicted", "target"}:
        raise ValueError("training.physics_frequency_source must be 'predicted' or 'target'.")
    extensions["physics_frequency_source"] = frequency_source
    checkpoint_epochs = tuple(sorted({int(value) for value in training.get("checkpoint_epochs", []) if int(value) > 0}))
    gradient_audit_epochs = tuple(sorted({int(value) for value in training.get("gradient_audit_epochs", []) if int(value) > 0}))
    select_model = str(training.get("select_model", "best_val")).lower()
    if select_model not in {"best_val", "final_epoch"}:
        raise ValueError("training.select_model must be 'best_val' or 'final_epoch'.")
    history = _read_history(out / "history.csv") if resumed_from_checkpoint else []
    gradient_audit = _read_csv_rows(out / "frequency_gradient_audit.csv") if resumed_from_checkpoint else []
    best, stale = _history_best_and_stale(history)
    best_path = out / "best_val.pt"; patience = int(training.get("early_stopping_patience", 30))
    start_epoch = int(resumed_checkpoint_epoch or 0) + 1
    if resumed_from_checkpoint:
        if resume_payload is not None and "optimizer_state_dict" in resume_payload and "scheduler_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"]); scheduler.load_state_dict(resume_payload["scheduler_state_dict"]); optimizer_state_restored = True
        else:
            resumed_lr = base_lr * _learning_rate_scale(int(resumed_checkpoint_epoch), epochs, warmup_lr, minimum_lr / base_lr)
            for group in optimizer.param_groups:
                group["lr"] = resumed_lr
            scheduler.last_epoch = int(resumed_checkpoint_epoch)
            scheduler._last_lr = [resumed_lr]
        logger.info("Resuming field-eigen training from %s at epoch=%d optimizer_state_restored=%s", resumed_checkpoint_path, resumed_checkpoint_epoch, optimizer_state_restored)

    def loader_for(indices: np.ndarray, *, shuffle: bool):
        return DataLoader(TensorDataset(*(value[indices] for value in tensors)), batch_size=int(training.get("batch_size", 16)), shuffle=shuffle, pin_memory=device_info.pin_memory)

    dense_loader = None
    dense_source_grid = None
    if dense_frequency is not None:
        dense_z = ((np.log(dense_frequency.frequency.astype(np.float64)) - frequency_mean) / frequency_std).astype(np.float32)
        dense_loader = DataLoader(
            TensorDataset(torch.as_tensor(dense_frequency.source_index), torch.as_tensor(dense_frequency.context), torch.as_tensor(dense_z)),
            batch_size=int(dict(training.get("dense_frequency_supervision") or {}).get("batch_size", training.get("batch_size", 16))),
            shuffle=True,
            pin_memory=device_info.pin_memory,
        )
        dense_source_grid = torch.as_tensor(dense_frequency.source_grid)

    if train:
        train_loader = loader_for(train_index, shuffle=True); val_loader = loader_for(data.indices["val"], shuffle=False)
        for epoch in range(start_epoch, epochs + 1):
            model.train(); epoch_data=[]; epoch_physics=[]; epoch_norm=[]; epoch_dense=[]; epoch_rayleigh=[]; active_lambda=_lambda_at_epoch(epoch, lambda_target, physics_warmup, physics_ramp) if regime == "pino_adapted" else 0.0
            audit_groups = _parameter_groups_for_gradient_audit(model) if epoch in gradient_audit_epochs and active_lambda > 0.0 else None
            dense_iterator = iter(dense_loader) if dense_loader is not None else None
            for batch_index, (grid, context, target, target_frequency, epsilon, k_point, dx, dy) in enumerate(train_loader):
                grid, context, target, target_frequency = grid.to(device), context.to(device), target.to(device), target_frequency.to(device)
                optimizer.zero_grad(set_to_none=True); predicted, predicted_frequency_z = _model_forward(model, str(model_cfg["name"]), grid, context)
                data_loss, _, _, _ = _data_loss(torch, predicted, predicted_frequency_z, target, target_frequency, epsilon=epsilon.to(device), loss_weights=loss_weights)
                dense_loss = rayleigh_loss = physics_loss = normalization_loss = torch.zeros((), device=device)
                if dense_iterator is not None and dense_source_grid is not None:
                    try:
                        dense_source_index, dense_context, dense_target_frequency = next(dense_iterator)
                    except StopIteration:
                        dense_iterator = iter(dense_loader); dense_source_index, dense_context, dense_target_frequency = next(dense_iterator)
                    _, dense_prediction_z = _model_forward(model, str(model_cfg["name"]), dense_source_grid[dense_source_index].to(device), dense_context.to(device))
                    dense_loss = (dense_prediction_z - dense_target_frequency.to(device)).square().mean()
                    data_loss = data_loss + float(loss_weights["frequency"]) * dense_loss
                if active_lambda > 0.0:
                    active = _torch_active_complex(torch, predicted)
                    predicted_physical_frequency = torch.exp(
                        physics_frequency_from_prediction(predicted_frequency_z, mode=frequency_gradient_mode)
                        * frequency_std
                        + frequency_mean
                    )
                    residual_frequency_z = physics_frequency_for_residual(
                        predicted_frequency_z,
                        target_frequency,
                        source=frequency_source,
                        gradient_mode=frequency_gradient_mode,
                    )
                    residual_physical_frequency = torch.exp(residual_frequency_z * frequency_std + frequency_mean)
                    residual = evaluate_te_bloch_eigen_residual_torch(active[..., :2], epsilon=epsilon.to(device), eigenfrequency=residual_physical_frequency, dx=dx.to(device), dy=dy.to(device), bloch_k_point=k_point.to(device))
                    combined_physics, physics_loss, normalization_loss = pino_te_physics_loss(residual, solver_floor_scale=solver_floor_factors[data.field_shape[0]])
                    high_resolutions = tuple(value for value in pde_resolutions if value != data.field_shape[0])
                    if high_resolutions:
                        high_resolution = high_resolutions[(epoch + batch_index) % len(high_resolutions)]
                        high_grid = _periodic_grid_torch(torch, epsilon.to(device), resolution=high_resolution, epsilon_range=tuple(float(value) for value in dict(raw["normalization"])["epsilon_range"]))
                        high_prediction, high_frequency_z = _model_forward(model, str(model_cfg["name"]), high_grid, context)
                        high_active = _torch_active_complex(torch, high_prediction)
                        high_predicted_frequency = torch.exp(physics_frequency_from_prediction(high_frequency_z, mode=frequency_gradient_mode) * frequency_std + frequency_mean)
                        high_residual_frequency_z = physics_frequency_for_residual(
                            high_frequency_z,
                            target_frequency,
                            source=frequency_source,
                            gradient_mode=frequency_gradient_mode,
                        )
                        high_residual_frequency = torch.exp(high_residual_frequency_z * frequency_std + frequency_mean)
                        high_dx = torch.full_like(dx.to(device), 1.0 / float(high_resolution)); high_dy = torch.full_like(dy.to(device), 1.0 / float(high_resolution))
                        high_residual = evaluate_te_bloch_eigen_residual_torch(high_active[..., :2], epsilon=epsilon.to(device), eigenfrequency=high_residual_frequency, dx=high_dx, dy=high_dy, bloch_k_point=k_point.to(device))
                        high_combined, high_physics, high_normalization = pino_te_physics_loss(high_residual, solver_floor_scale=solver_floor_factors[high_resolution])
                        combined_physics = (combined_physics + 2.0 * high_combined) / 3.0
                        physics_loss = (physics_loss + 2.0 * high_physics) / 3.0
                        normalization_loss = (normalization_loss + 2.0 * high_normalization) / 3.0
                        if rayleigh_weight > 0.0:
                            rayleigh_loss = (_rayleigh_consistency_loss(torch, physical_frequency=predicted_physical_frequency, residual=residual) + 2.0 * _rayleigh_consistency_loss(torch, physical_frequency=high_predicted_frequency, residual=high_residual)) / 3.0
                    elif rayleigh_weight > 0.0:
                        rayleigh_loss = _rayleigh_consistency_loss(torch, physical_frequency=predicted_physical_frequency, residual=residual)
                    pde_loss = active_lambda * combined_physics
                    loss = data_loss + pde_loss + rayleigh_weight * rayleigh_loss
                    if audit_groups is not None and batch_index == 0:
                        audit_row = {"epoch": epoch, "batch_index": batch_index, "physics_frequency_gradient": frequency_gradient_mode, "physics_frequency_source": frequency_source, "pde_loss": float(pde_loss.detach().cpu()), **_pde_gradient_audit(torch, pde_loss, audit_groups)}
                        gradient_audit.append(audit_row)
                        _write_csv(out / "frequency_gradient_audit.csv", gradient_audit)
                else:
                    loss = data_loss
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0))); optimizer.step()
                epoch_data.append(float(data_loss.detach().cpu())); epoch_physics.append(float(physics_loss.detach().cpu())); epoch_norm.append(float(normalization_loss.detach().cpu())); epoch_dense.append(float(dense_loss.detach().cpu())); epoch_rayleigh.append(float(rayleigh_loss.detach().cpu()))
            scheduler.step(); model.eval(); validation=[]
            with torch.no_grad():
                for grid, context, target, target_frequency, epsilon, *_ in val_loader:
                    predicted, predicted_frequency_z = _model_forward(model, str(model_cfg["name"]), grid.to(device), context.to(device)); value, *_ = _data_loss(torch, predicted, predicted_frequency_z, target.to(device), target_frequency.to(device), epsilon=epsilon.to(device)); validation.append(float(value.cpu()))
            val_loss = float(np.mean(validation)); history.append({"epoch": epoch, "train_data_loss": float(np.mean(epoch_data)), "train_dense_frequency_loss": float(np.mean(epoch_dense)), "train_physics_loss": float(np.mean(epoch_physics)), "train_normalization_loss": float(np.mean(epoch_norm)), "train_rayleigh_consistency_loss": float(np.mean(epoch_rayleigh)), "val_data_loss": val_loss, "lambda_pde_active": active_lambda, "learning_rate": float(optimizer.param_groups[0]["lr"])})
            payload = _checkpoint_payload(model=model, model_cfg=model_cfg, frequency_mean=frequency_mean, frequency_std=frequency_std, dataset_hash=data.dataset_hash, seed=seed, regime=regime, lambda_pde=lambda_target, epoch=epoch, validation_loss=val_loss, deterministic=deterministic, loss_weights=loss_weights, physics_frequency_gradient=frequency_gradient_mode, physics_frequency_source=frequency_source, training_extensions=extensions, optimizer=optimizer, scheduler=scheduler)
            if val_loss < best:
                best = val_loss; stale = 0
                torch.save(payload, best_path)
            else:
                stale += 1
            if epoch in checkpoint_epochs:
                torch.save(payload, out / f"checkpoint_epoch_{epoch}.pt")
            if patience > 0 and stale >= patience:
                break
            _write_csv(out / "history.csv", history)
            logger.info("epoch=%d val_data_loss=%.8f best=%.8f stale=%d", epoch, val_loss, best, stale)
        if select_model == "best_val":
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["state_dict"])

    split_name = str(dict(raw.get("evaluation") or {}).get("split", "test")); evaluation_index = data.indices[split_name]
    predictions=[]; frequencies=[]; evaluation_started=time.perf_counter(); model.eval()
    with torch.no_grad():
        for grid, context, *_ in loader_for(evaluation_index, shuffle=False):
            predicted, predicted_frequency_z = _model_forward(model, str(model_cfg["name"]), grid.to(device), context.to(device)); predictions.append(predicted.cpu().numpy()); frequencies.append(torch.exp(predicted_frequency_z * frequency_std + frequency_mean).cpu().numpy())
        synchronize_device(device, torch_module=torch)
    evaluation_time = time.perf_counter() - evaluation_started
    prediction = np.concatenate(predictions); physical_frequency = np.concatenate(frequencies)
    metrics, rows = _evaluate_metrics(data, prediction, physical_frequency, evaluation_index)
    metrics.update({"train_time_s": float(evaluation_started - started) if train else 0.0, "eval_time_s": evaluation_time, "inference_time_per_sample_ms": 1000.0 * evaluation_time / max(len(evaluation_index), 1), "wall_time_total_s": time.perf_counter() - started, "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())), "peak_gpu_memory_mb": peak_gpu_memory_mb(device, torch_module=torch), "n_train": len(data.indices["train"]), "n_val": len(data.indices["val"]), "n_test": len(data.indices["test"]), "seed": seed, "lambda_pde": lambda_target, "training_regime": regime, "physics_frequency_source": frequency_source, "epochs_completed": len(history), "resumed_from_checkpoint": resumed_from_checkpoint, "resumed_checkpoint_epoch": resumed_checkpoint_epoch, "resumed_checkpoint_path": str(resumed_checkpoint_path) if resumed_checkpoint_path else None, "optimizer_state_restored": optimizer_state_restored})
    checkpoint_payload = _checkpoint_payload(model=model, model_cfg=model_cfg, frequency_mean=frequency_mean, frequency_std=frequency_std, dataset_hash=data.dataset_hash, seed=seed, regime=regime, lambda_pde=lambda_target, epoch=len(history) if train else None, validation_loss=history[-1]["val_data_loss"] if history else None, deterministic=deterministic, loss_weights=loss_weights, physics_frequency_gradient=frequency_gradient_mode, physics_frequency_source=frequency_source, training_extensions=extensions, optimizer=optimizer, scheduler=scheduler)
    torch.save(checkpoint_payload, out / "model.pt")
    _write_csv(out / "history.csv", history); _write_csv(out / "per_case_metrics.csv", rows); _write_csv(out / "frequency_gradient_audit.csv", gradient_audit)
    write_json(out / "field_component_spec.json", {"polarization": "TE", "predicted_complex_components": ["Ex", "Ey", "Hz"], "channel_order": ["Ex.real", "Ey.real", "Hz.real", "Ex.imag", "Ey.imag", "Hz.imag"], "reconstructed_zero_components": ["Ez", "Hx", "Hy"], "frequency_is_input": False})
    write_resolved_config(out, yaml.safe_dump({key: value for key, value in raw.items() if key != "_path"}, sort_keys=False, allow_unicode=True))
    write_metrics(out, metrics); write_json(out / "model_identity.json", describe_model_identity(model, str(model_cfg["name"]), spec=get_model_spec(str(model_cfg["name"]))))
    status = "pino_field_adapted_pilot" if regime == "pino_adapted" else "supervised_field_baseline_v2"
    write_results(out, {"status": status, "stage": "train" if train else "eval", "model_name": model_cfg["name"], "training_regime": regime, "checkpoint_path": str(out / "model.pt"), "dataset_hash": data.dataset_hash, "architecture_fingerprint": architecture_fingerprint(model), "lambda_pde": lambda_target, "loss_weights": loss_weights, "physics_frequency_gradient": frequency_gradient_mode, "physics_frequency_source": frequency_source, "training_extensions": extensions, "metrics": metrics})
    write_run_meta(out, {**collect_device_metadata(device_info), "status": status, "resumed_from_checkpoint": resumed_from_checkpoint, "resumed_checkpoint_epoch": resumed_checkpoint_epoch, "optimizer_state_restored": optimizer_state_restored}); logger.info("Completed B3 field run %s", out)
    return out


def run_field_eigen_v2_train(path: str | Path) -> Path:
    return _run(_read(path), train=True)


def run_field_eigen_v2_eval(path: str | Path) -> Path:
    raw = _read(path)
    checkpoint = Path(str(dict(raw["model"])["checkpoint_path"]))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    return _run(raw, train=False, checkpoint=checkpoint)


def run_field_eigen_v2_benchmark(path: str | Path, *, run_dir: Path | None = None) -> Path:
    raw = _read(path); benchmark = dict(raw.get("benchmark") or {})
    parent = make_run_dir(str(dict(raw["experiment"])["experiment_id"]), runs_dir=ROOT / str(dict(raw.get("output") or {}).get("runs_dir", "runs")), run_dir=run_dir)
    logger = setup_logging(log_file=parent / "run.log", run_meta={"stage": "field_eigen_v2_benchmark"}); ensure_figures_dir(parent)
    cases = list(benchmark.get("cases") or [])
    if not cases:
        cases = [{"name": f"{model}_seed{seed}", "model": {"name": model, "params": dict(benchmark.get("model_params") or {}).get(model, {})}, "training": {"seed": seed}} for model in benchmark.get("models", []) for seed in dict(raw["training"]).get("seeds", [0])]
    rows=[]; case_configs: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_name = str(case["name"]); case_raw = {**raw, "model": {**dict(raw.get("model") or {}), **dict(case.get("model") or {})}, "training": {**dict(raw["training"]), **dict(case.get("training") or {})}, "representation": {**dict(raw.get("representation") or {}), **dict(case.get("representation") or {})}}
        case_configs[case_name] = case_raw
        case_train = bool(case.get("train", True))
        checkpoint = None
        if not case_train:
            checkpoint_value = case.get("checkpoint_path")
            if not checkpoint_value:
                raise ValueError(f"Evaluation-only benchmark case {case_name!r} requires checkpoint_path.")
            checkpoint = Path(str(checkpoint_value))
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
        case_dir = parent / "cases" / case_name
        if (case_dir / "metrics.json").is_file() and (case_dir / "model.pt").is_file():
            run = case_dir; logger.info("Reusing completed B3 case %s", case_name)
        else:
            run = _run(case_raw, run_dir=case_dir, train=case_train, checkpoint=checkpoint)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8")); rows.append({"case": case_name, "model": case_raw["model"]["name"], "seed": case_raw["training"].get("seed", 0), "regime": case_raw["training"].get("regime", "data_only"), "lambda_pde": case_raw["training"].get("lambda_pde", 0.0), "frequency_weight": dict(case_raw["training"].get("loss_weights") or {}).get("frequency", 0.1), "physics_frequency_gradient": case_raw["training"].get("physics_frequency_gradient", "full"), "physics_frequency_source": case_raw["training"].get("physics_frequency_source", "predicted"), "b4_audited": bool(case_raw.get("b4_audited", case_train)), "run_dir": str(run), "execution": "train" if case_train else "frozen_checkpoint_eval", "checkpoint_source": str(checkpoint) if checkpoint else None, **{key: metrics[key] for key in ("e_relative_l2", "h_relative_l2", "frequency_mre_pct", "maxwell_relative_residual", "boundary_relative_residual", "parameter_count", "train_time_s")}}); logger.info("Completed B3 case %s", case_name)
    selection = None
    if str(benchmark.get("selection_rule", "")).startswith("pino_validation_residual"):
        references = [row for row in rows if row["regime"] == "data_only"]
        if len(references) != 1:
            raise ValueError("PINO lambda selection requires exactly one data-only validation reference.")
        reference = references[0]
        eligible = [row for row in rows if row["regime"] == "pino_adapted" and float(row["e_relative_l2"]) <= 1.02 * float(reference["e_relative_l2"])]
        selected = min(eligible, key=lambda row: (float(row["maxwell_relative_residual"]), float(row["e_relative_l2"]))) if eligible else None
        selection = {"status": "lambda_selected" if selected else "blocked", "rule": benchmark["selection_rule"], "data_only_reference": reference, "eligible_case_count": len(eligible), "selected_case": selected, "selected_lambda_pde": float(selected["lambda_pde"]) if selected else None}
        write_json(parent / "lambda_selection.json", selection)
    elif str(benchmark.get("selection_rule", "")) == "pino_frequency_guard_validation":
        references = [row for row in rows if row["regime"] == "data_only"]
        if len(references) != 1:
            raise ValueError("B4 frequency guard requires exactly one data-only FNO validation reference.")
        reference = references[0]
        candidates = [row for row in rows if row["regime"] == "pino_adapted" and bool(row["b4_audited"])]
        eligible = [
            row for row in candidates
            if float(row["frequency_mre_pct"]) <= 1.02 * float(reference["frequency_mre_pct"])
            and float(row["e_relative_l2"]) <= 1.02 * float(reference["e_relative_l2"])
            and float(row["maxwell_relative_residual"]) <= 0.80 * float(reference["maxwell_relative_residual"])
            and all(math.isfinite(float(row[key])) for key in ("frequency_mre_pct", "e_relative_l2", "maxwell_relative_residual", "boundary_relative_residual"))
        ]
        selected = min(eligible, key=lambda row: (float(row["maxwell_relative_residual"]), float(row["e_relative_l2"]), float(row["frequency_mre_pct"]))) if eligible else None
        selection = {
            "status": "frequency_guard_selected" if selected else "blocked", "rule": "pino_frequency_guard_validation",
            "selection_split": str(dict(raw.get("evaluation") or {}).get("split", "val")), "data_only_reference": reference,
            "candidate_count": len(candidates), "eligible_case_count": len(eligible), "selected_case": selected,
            "thresholds": {"frequency_mre_factor": 1.02, "e_relative_l2_factor": 1.02, "residual_reduction_pct": 20.0},
        }
        write_json(parent / "frequency_guard_selection.json", selection)
        if selected:
            write_json(parent / "frozen_candidate.json", {"selection": selection, "checkpoint_path": str(Path(selected["run_dir"]) / "model.pt"), "checkpoint_sha256": _hash(Path(selected["run_dir"]) / "model.pt")})
            formal_rows: list[dict[str, Any]] = []
            for formal_seed in tuple(int(value) for value in benchmark.get("formal_selected_seeds", [])):
                formal_raw = {**case_configs[str(selected["case"])], "evaluation": {**dict(raw.get("evaluation") or {}), "split": "test"}, "training": {**dict(case_configs[str(selected["case"])] ["training"]), "seed": formal_seed}}
                formal_dir = parent / "frozen_candidate_three_seed" / f"seed{formal_seed}"
                formal_run = _run(formal_raw, run_dir=formal_dir, train=True)
                formal_metrics = json.loads((formal_run / "metrics.json").read_text(encoding="utf-8"))
                formal_rows.append({"seed": formal_seed, "checkpoint_path": str(formal_run / "model.pt"), "checkpoint_sha256": _hash(formal_run / "model.pt"), **{key: formal_metrics[key] for key in ("e_relative_l2", "h_relative_l2", "frequency_mre_pct", "maxwell_relative_residual", "boundary_relative_residual")}})
            _write_csv(parent / "frozen_candidate_three_seed.csv", formal_rows)
            write_json(parent / "frozen_candidate_manifest.json", {"selection": selection, "evaluation_split": "test", "cases": formal_rows, "model": dict(raw["model"]), "dataset_hash": _hash(ROOT / str(dict(raw["dataset"])["path"]))})
    _write_csv(parent / str(benchmark.get("table_filename", "leaderboard.csv")), rows); _write_csv(parent / "ablation.csv", rows); _write_csv(parent / "history.csv", rows)
    write_resolved_config(parent, yaml.safe_dump({key: value for key, value in raw.items() if key != "_path"}, sort_keys=False, allow_unicode=True)); write_metrics(parent, {"case_count": len(rows), "cases": rows, "selection": selection}); write_results(parent, {"status": str(dict(raw["experiment"]).get("stage", "field_eigen_v2_benchmark")), "case_count": len(rows), "table": str(benchmark.get("table_filename", "leaderboard.csv")), "selection": selection}); write_run_meta(parent, {"status": str(dict(raw["experiment"]).get("stage", "field_eigen_v2_benchmark"))})
    return parent


__all__ = ["FieldEigenData", "active_channels_to_complex", "complex_to_active_channels", "is_field_eigen_v2_config", "load_field_eigen_v2", "phase_invariant_field_loss", "physics_frequency_for_residual", "reconstruct_full_te_fields", "run_field_eigen_v2_benchmark", "run_field_eigen_v2_eval", "run_field_eigen_v2_train"]
