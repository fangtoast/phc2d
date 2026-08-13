"""Generative inverse CLI runtime for smoke sampling, ranking, and replay."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.data import load_forward_dataset
from phc2d.inverse.candidates import CandidateBatch, candidate_batch_metrics, separated_candidate_indices
from phc2d.inverse.config import GenerativeInverseConfig, load_inverse_run_config
from phc2d.inverse.generator_models import (
    _ConditionalGenerator as _ConditionalGenerator,
    _DirectInverseGenerator as _DirectInverseGenerator,
    _MixtureConditionalGenerator as _MixtureConditionalGenerator,
    _generator_kind,
    _make_generator_network,
)
from phc2d.inverse.generator_runtime import _MockConditionalGenerator, _TrainedConditionalGenerator
from phc2d.inverse.morphology import candidate_morphology_metrics
from phc2d.inverse.objective import (
    cross_surrogate_consistency_loss,
    gap_descriptor_loss,
    objective_loss_components,
)
from phc2d.inverse.runtime_support import (
    _as_mapping,
    _read_csv_rows,
    _read_yaml as _read_yaml,
    _to_jsonable,
)
from phc2d.inverse.surrogate_runtime import (
    _ForwardFrozenSurrogate,
    _MockSurrogate,
    _QueryFrozenSurrogate,
    _resolve_forward_surrogate_checkpoint as _resolve_forward_surrogate_checkpoint,
    _resolve_query_surrogate_checkpoint as _resolve_query_surrogate_checkpoint,
)
from phc2d.plot import plot_metric_report
from phc2d.runs import (
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    write_metrics,
    write_run_meta,
    write_resolved_config,
    write_results,
)
from phc2d.training import (
    ResolvedDeviceInfo,
    collect_device_metadata,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    resolve_device,
    synchronize_device,
)


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(str(item) for item in (base_seed, *parts))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Inverse neural sampling/ranking requires optional dependency 'torch'. "
            "Install the train extra or run in WSL Ubuntu conda meep_env."
        ) from exc
    return torch


def _configured_learning_rates(cfg: GenerativeInverseConfig) -> tuple[float, ...]:
    return tuple(float(item) for item in cfg.learning_rates) or (float(cfg.learning_rate),)


def _learning_rate_label(value: float) -> str:
    return f"lr{float(value):.0e}".replace("+", "").replace("-", "m")


def _seed_lr_suffix(cfg: GenerativeInverseConfig, seed: int) -> str:
    suffix = f"seed{int(seed)}"
    if len(_configured_learning_rates(cfg)) > 1:
        suffix = f"{suffix}_{_learning_rate_label(cfg.learning_rate)}"
    return suffix


def _make_inverse_optimizer(cfg: GenerativeInverseConfig, parameters: Any, *, torch: Any) -> Any:
    name = str(cfg.optimizer).lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    if name == "adam":
        return torch.optim.Adam(parameters, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    raise ValueError(f"Unsupported inverse optimizer: {cfg.optimizer!r}")


def _resolve_runtime_device(cfg: GenerativeInverseConfig, torch: Any) -> ResolvedDeviceInfo:
    return resolve_device(cfg.runtime, torch_module=torch)


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


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _torch_save_atomic(payload: Mapping[str, Any], path: Path, *, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_targets(
    cfg: GenerativeInverseConfig,
) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]]]:
    bundle = load_forward_dataset(cfg.dataset_path)
    split = str(cfg.target_split).strip().lower()
    if split in {"train", "idx_train"}:
        indices = bundle.idx_train
    elif split in {"val", "validation", "idx_val"}:
        indices = bundle.idx_val
    elif split in {"test", "idx_test"}:
        indices = bundle.idx_test
    else:
        raise ValueError(f"Unknown target split '{cfg.target_split}'")
    if len(indices) == 0:
        raise ValueError(f"No data for split '{split}' in {cfg.dataset_path}")
    indices = np.asarray(indices, dtype=np.int64)
    split_indices = indices.copy()
    if cfg.target_manifest_path is not None:
        manifest_path = Path(cfg.target_manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_indices = payload.get("target_indices")
        if not isinstance(manifest_indices, list) or not manifest_indices:
            raise ValueError(f"target manifest has no target_indices: {manifest_path}")
        indices = np.asarray([int(item) for item in manifest_indices], dtype=np.int64)
        if int(np.unique(indices).size) != int(indices.size):
            raise ValueError(f"target manifest contains duplicate indices: {manifest_path}")
        split_set = {int(item) for item in split_indices}
        if any(int(item) not in split_set for item in indices):
            raise ValueError(
                f"target manifest contains an index outside split '{split}': {manifest_path}"
            )
        manifest_split = str(payload.get("dataset_split", "")).strip().lower()
        if manifest_split and manifest_split not in {split, "validation" if split == "val" else split}:
            raise ValueError(
                f"target manifest split mismatch: manifest={manifest_split!r}, config={split!r}"
            )
    elif cfg.target_limit is not None and int(cfg.target_limit) < int(indices.size):
        limit = int(cfg.target_limit)
        if cfg.target_selection_seed is None:
            indices = indices[:limit]
        else:
            rng = np.random.default_rng(int(cfg.target_selection_seed))
            selected = rng.choice(indices, size=limit, replace=False)
            indices = np.asarray(sorted(int(item) for item in selected), dtype=np.int64)
    if cfg.target_limit is not None and int(indices.size) > int(cfg.target_limit):
        raise ValueError(
            "target_limit cannot truncate an explicit target manifest; "
            "use a manifest with the intended target count"
        )
    candidate_dim = int(np.asarray(bundle.rho, dtype=np.float32).reshape(int(bundle.rho.shape[0]), -1).shape[1])
    dataset_id = Path(cfg.dataset_path).stem
    target_metadata: list[dict[str, Any]] = []
    for index in indices:
        idx = int(index)
        metadata = dict(bundle.metadata[idx]) if idx < len(bundle.metadata) and isinstance(bundle.metadata[idx], dict) else {}
        metadata.setdefault("dataset_id", dataset_id)
        target_metadata.append(metadata)
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(bundle.Wh, dtype=np.float32),
        int(candidate_dim),
        target_metadata,
    )


def _condition_seed(cfg: GenerativeInverseConfig, target_id: str, index: int) -> int:
    return _stable_seed(cfg.seed + cfg.candidate_seed_offset, "generator", target_id, index)


def _surrogate_seed(cfg: GenerativeInverseConfig, target_id: str, index: int) -> int:
    return _stable_seed(cfg.seed + cfg.surrogate_seed_offset, "surrogate", target_id, index)


def _replay_seed(cfg: GenerativeInverseConfig, target_id: str, index: int) -> int:
    return _stable_seed(cfg.seed + cfg.replay_seed_offset, "replay", target_id, index)


def _target_to_condition_dict(cfg: GenerativeInverseConfig, target_id: str, target: np.ndarray) -> dict[str, Any]:
    return {cfg.condition_target: _to_jsonable(target), "target_id": target_id}


def _target_id(cfg: GenerativeInverseConfig, target_index: int, local_pos: int) -> str:
    return f"{cfg.experiment_id}_seed{int(cfg.seed)}_{cfg.condition_target}_{int(target_index):05d}_{local_pos:03d}"


def _apply_filters(
    batch: CandidateBatch,
    cfg: GenerativeInverseConfig,
) -> tuple[CandidateBatch, dict[str, float]]:
    arr = batch.candidates
    mins, maxs = cfg.candidate_value_range
    within = np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))
    within &= (arr >= float(mins)).all(axis=tuple(range(1, arr.ndim)))
    within &= (arr <= float(maxs)).all(axis=tuple(range(1, arr.ndim)))

    min_fill = float(cfg.morphology_filters.get("min_fill_fraction", 0.0))
    max_fill = float(cfg.morphology_filters.get("max_fill_fraction", 1.0))
    fill = np.array([float(np.mean(np.asarray(item) >= cfg.threshold)) for item in arr], dtype=np.float32)
    within &= (fill >= min_fill) & (fill <= max_fill)

    if not bool(np.any(within)):
        within = np.zeros(arr.shape[0], dtype=bool)
        within[0] = True

    idx = np.where(within)[0]
    separation_cfg = _as_mapping(cfg.candidate_separation)
    separation_enabled = bool(separation_cfg.get("enabled", False))
    deduplicate_binary = bool(separation_cfg.get("deduplicate_binary", False))
    raw_min_hamming = separation_cfg.get("min_hamming", 0.0)
    if isinstance(raw_min_hamming, str):
        normalized = raw_min_hamming.strip().lower()
        if normalized in {
            "auto",
            "auto_1_over_dim",
            "auto_1_over_geometry_dof_count",
            "one_bit",
            "1/candidate_dim",
        }:
            min_hamming = 1.0 / float(max(int(arr.reshape(int(arr.shape[0]), -1).shape[1]), 1))
        else:
            raise ValueError(f"Unsupported inverse.candidate_separation.min_hamming: {raw_min_hamming!r}")
    else:
        min_hamming = float(raw_min_hamming or 0.0)
    if separation_enabled or deduplicate_binary or min_hamming > 0.0:
        local_idx = separated_candidate_indices(
            arr[idx],
            threshold=cfg.threshold,
            deduplicate_binary=deduplicate_binary or separation_enabled,
            min_hamming=min_hamming,
        )
        idx = idx[local_idx]
    filtered = CandidateBatch(
        target_id=batch.target_id,
        condition=batch.condition,
        rho_candidates=arr[idx] if batch.candidate_kind == "rho_candidates" else None,
        grid_candidates=arr[idx] if batch.candidate_kind == "grid_candidates" else None,
        input_repr=batch.input_repr,
        metadata=[dict(item) for item in [batch.metadata[int(i)] for i in idx]],
    )
    metrics = {
        "filter_pass_ratio": float(idx.size / max(arr.shape[0], 1)),
        "candidate_count_raw": float(arr.shape[0]),
        "candidate_count_filtered": float(filtered.n_candidates),
        "candidate_separation_enabled": float(bool(separation_enabled or deduplicate_binary or min_hamming > 0.0)),
        "candidate_separation_deduplicate_binary": float(bool(deduplicate_binary or separation_enabled)),
        "candidate_separation_min_hamming": float(min_hamming),
    }
    return filtered, metrics


def _relative_mre(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    tgt = np.broadcast_to(tgt, pred.shape)
    return np.abs(pred - tgt) / np.maximum(np.abs(tgt), 1.0e-8)


@lru_cache(maxsize=16)
def _dataset_band_count(dataset_path: str) -> int:
    return int(getattr(load_forward_dataset(Path(dataset_path)), "nb", 0) or 0)


def _surrogate_error_tensor(predictions: Any, target: Any, cfg: GenerativeInverseConfig, torch: Any) -> tuple[Any, dict[str, float]]:
    metric = str(cfg.ranking.get("error_metric", "full_mre")).lower()
    flat_pred = predictions.reshape(int(predictions.shape[0]), -1)
    target_broadcast = target.reshape(1, -1).expand_as(flat_pred)
    errors = torch.abs(flat_pred - target_broadcast) / torch.clamp(torch.abs(target_broadcast), min=1.0e-8)
    full_errors = errors.mean(dim=1)

    metrics: dict[str, float] = {
        "best_of_k_surrogate_mre_full": float(full_errors.min().detach().cpu().item())
        if int(full_errors.numel())
        else 0.0,
    }
    if metric not in {"bands_2_10_mre", "avg_mre_bands_2_10_pct", "bands2_10_mre"}:
        return full_errors, metrics

    nb = _dataset_band_count(str(cfg.dataset_path))
    if nb < 2 or int(flat_pred.shape[1]) % nb != 0:
        metrics["ranking_metric_fallback_full_mre"] = 1.0
        return full_errors, metrics

    reshaped_pred = flat_pred.reshape(int(flat_pred.shape[0]), -1, nb)
    reshaped_target = target_broadcast.reshape(int(flat_pred.shape[0]), -1, nb)
    band_pred = reshaped_pred[:, :, 1:]
    band_target = reshaped_target[:, :, 1:]
    band_errors = torch.abs(band_pred - band_target) / torch.clamp(torch.abs(band_target), min=1.0e-8)
    candidate_errors = band_errors.reshape(int(flat_pred.shape[0]), -1).mean(dim=1)
    metrics["best_of_k_surrogate_mre_bands_2_10"] = float(candidate_errors.min().detach().cpu().item())
    metrics["ranking_band_range_start_one_based"] = 2.0
    metrics["ranking_band_range_end_one_based"] = float(nb)
    return candidate_errors, metrics


def _filter_candidate_tensor(raw: Any, cfg: GenerativeInverseConfig, torch: Any) -> tuple[Any, dict[str, float], np.ndarray]:
    lo, hi = cfg.candidate_value_range
    flat = raw.reshape(int(raw.shape[0]), -1)
    within = torch.isfinite(flat).all(dim=1)
    within = within & (flat >= float(lo)).all(dim=1) & (flat <= float(hi)).all(dim=1)
    fill = (flat >= float(cfg.threshold)).to(dtype=torch.float32).mean(dim=1)
    min_fill = float(cfg.morphology_filters.get("min_fill_fraction", 0.0))
    max_fill = float(cfg.morphology_filters.get("max_fill_fraction", 1.0))
    within = within & (fill >= min_fill) & (fill <= max_fill)
    if not bool(torch.any(within).item()):
        within = torch.zeros_like(within, dtype=torch.bool)
        within[0] = True
    idx = torch.nonzero(within, as_tuple=False).flatten()
    filtered = raw.index_select(0, idx)
    separation_cfg = _as_mapping(cfg.candidate_separation)
    separation_enabled = bool(separation_cfg.get("enabled", False))
    deduplicate_binary = bool(separation_cfg.get("deduplicate_binary", False))
    raw_min_hamming = separation_cfg.get("min_hamming", 0.0)
    if isinstance(raw_min_hamming, str):
        normalized = raw_min_hamming.strip().lower()
        if normalized in {
            "auto",
            "auto_1_over_dim",
            "auto_1_over_geometry_dof_count",
            "one_bit",
            "1/candidate_dim",
        }:
            min_hamming = 1.0 / float(max(int(flat.shape[1]), 1))
        else:
            raise ValueError(f"Unsupported inverse.candidate_separation.min_hamming: {raw_min_hamming!r}")
    else:
        min_hamming = float(raw_min_hamming or 0.0)
    if separation_enabled or deduplicate_binary or min_hamming > 0.0:
        local_idx_np = separated_candidate_indices(
            filtered.detach().cpu().numpy(),
            threshold=cfg.threshold,
            deduplicate_binary=deduplicate_binary or separation_enabled,
            min_hamming=min_hamming,
        )
        local_idx = torch.as_tensor(local_idx_np, dtype=torch.long, device=idx.device)
        idx = idx.index_select(0, local_idx)
        filtered = filtered.index_select(0, local_idx)
    metrics = {
        "filter_pass_ratio": float(idx.numel() / max(int(raw.shape[0]), 1)),
        "candidate_count_raw": float(int(raw.shape[0])),
        "candidate_count_filtered": float(int(filtered.shape[0])),
        "candidate_separation_enabled": float(bool(separation_enabled or deduplicate_binary or min_hamming > 0.0)),
        "candidate_separation_deduplicate_binary": float(bool(deduplicate_binary or separation_enabled)),
        "candidate_separation_min_hamming": float(min_hamming),
    }
    return filtered, metrics, idx.detach().cpu().numpy().astype(np.int64)


def _find_latest_generator_checkpoint(cfg: GenerativeInverseConfig) -> Path | None:
    if cfg.run_dir is not None:
        run_dir_model = cfg.run_dir / "model.pt"
        if run_dir_model.is_file():
            return run_dir_model
        run_dir_seed = cfg.run_dir / f"model_seed{int(cfg.seed)}.pt"
        if run_dir_seed.is_file():
            return run_dir_seed
        run_dir_objective_seed = cfg.run_dir / f"objective_model_seed{int(cfg.seed)}.pt"
        if run_dir_objective_seed.is_file():
            return run_dir_objective_seed
    if cfg.generator_checkpoint_path is not None:
        return cfg.generator_checkpoint_path
    runs_root = cfg.runs_dir
    if not runs_root.exists():
        return None
    candidates: list[Path] = []
    for path in runs_root.rglob(f"model_seed{int(cfg.seed)}.pt"):
        if path.parent.name.startswith(cfg.experiment_id):
            candidates.append(path)
    for path in runs_root.rglob("model.pt"):
        if path.parent.name.startswith(cfg.experiment_id):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _find_latest_experiment_model_checkpoint(cfg: GenerativeInverseConfig) -> Path | None:
    runs_root = cfg.runs_dir
    if not runs_root.exists():
        return None
    candidates: list[Path] = []
    for name in ("model.pt", f"objective_model_seed{int(cfg.seed)}.pt", f"model_seed{int(cfg.seed)}.pt"):
        for path in runs_root.rglob(name):
            parent = path.parent.name
            if parent.startswith(cfg.experiment_id) and "_sample_" not in parent:
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _prepare_sample_config(cfg: GenerativeInverseConfig) -> GenerativeInverseConfig:
    if cfg.experiment_id.endswith("_sample"):
        return cfg
    if not str(cfg.stage).startswith("inverse_objective_train"):
        return cfg
    checkpoint = _find_latest_experiment_model_checkpoint(cfg) or cfg.generator_checkpoint_path
    return replace(
        cfg,
        experiment_id=f"{cfg.experiment_id}_sample",
        stage="inverse_sample_surrogate_side_no_mpb_meep",
        generator_checkpoint_path=checkpoint,
    )


def _build_generator(cfg: GenerativeInverseConfig, *, candidate_dim: int, torch: Any, device: Any) -> Any:
    checkpoint = _find_latest_generator_checkpoint(cfg)
    if checkpoint is not None and checkpoint.is_file():
        return _TrainedConditionalGenerator(cfg, checkpoint, torch=torch, device=device)
    if not cfg.allow_mock_generator:
        raise ValueError(
            "Formal inverse sampling requires a trained generator checkpoint. "
            "Run `phc2d inverse train --config ...` first or set inverse.generator_checkpoint_path."
        )
    return _MockConditionalGenerator(
        cfg=cfg,
        dim=candidate_dim,
        target_seed_anchor=0.0,
        seed=int(cfg.seed),
        torch=torch,
        device=device,
    )


def _build_surrogate(
    cfg: GenerativeInverseConfig,
    *,
    input_dim: int,
    target_shape: tuple[int, ...],
    torch: Any,
    device: Any,
) -> Any:
    surrogate_kind = _as_mapping(cfg.surrogate).get("kind")
    if surrogate_kind == "query":
        return _QueryFrozenSurrogate(cfg, torch=torch, device=device)
    if surrogate_kind == "forward":
        return _ForwardFrozenSurrogate(cfg, torch=torch, device=device)
    if not cfg.allow_mock_surrogate:
        raise ValueError(
            "Formal inverse ranking requires surrogate.kind=query or surrogate.kind=forward with a frozen checkpoint. "
            "Mock surrogate ranking is disabled by inverse.allow_mock_surrogate=false."
        )
    return _MockSurrogate(
        input_dim=input_dim,
        target_shape=target_shape,
        seed=int(cfg.seed),
        torch=torch,
        device=device,
    )


def _objective_secondary_surrogate_config(cfg: GenerativeInverseConfig, objective: Mapping[str, Any]) -> dict[str, Any]:
    secondary = _as_mapping(objective.get("secondary_surrogate"))
    if not secondary:
        secondary = _as_mapping(objective.get("cross_surrogate"))
    return secondary


def _build_objective_secondary_surrogate(
    cfg: GenerativeInverseConfig,
    objective: Mapping[str, Any],
    *,
    input_dim: int,
    target_shape: tuple[int, ...],
    torch: Any,
    device: Any,
) -> Any | None:
    secondary_cfg = _objective_secondary_surrogate_config(cfg, objective)
    cross_weight = float(objective.get("cross_surrogate_weight", 0.0) or 0.0)
    if not secondary_cfg:
        if cross_weight:
            raise ValueError(
                "cross-surrogate objective requires inverse.ranking.objective.secondary_surrogate "
                "with a resolvable frozen forward/query checkpoint"
            )
        return None
    secondary_parent = replace(cfg, surrogate=secondary_cfg)
    try:
        return _build_surrogate(
            secondary_parent,
            input_dim=input_dim,
            target_shape=target_shape,
            torch=torch,
            device=device,
        )
    except ValueError as exc:
        raise ValueError(
            "cross-surrogate objective could not build secondary surrogate; "
            "check inverse.ranking.objective.secondary_surrogate checkpoint fields"
        ) from exc


def _rank_candidates_on_device(
    *,
    candidates: Any,
    surrogate: Any,
    target: Any,
    cfg: GenerativeInverseConfig,
    torch: Any,
    device_info: ResolvedDeviceInfo,
) -> tuple[dict[str, Any], dict[str, float]]:
    device = device_info.torch_device
    synchronize_device(device, torch_module=torch)
    start = time.perf_counter()
    predictions = surrogate.predict_tensor(candidates)
    candidate_errors, metric_payload = _surrogate_error_tensor(predictions, target, cfg, torch)
    ranked = torch.argsort(candidate_errors, stable=True)
    selected = ranked[: max(1, min(int(cfg.surrogate_top_m), int(ranked.numel())))]
    synchronize_device(device, torch_module=torch)
    elapsed = time.perf_counter() - start
    selected_predictions = predictions.index_select(0, selected)
    selected_errors = candidate_errors.index_select(0, selected)
    best_prediction = selected_predictions[0].reshape(-1)
    flat_target = target.reshape(-1)
    best_delta = best_prediction - flat_target
    best_abs = torch.abs(best_delta)
    best_rel = best_abs / torch.clamp(torch.abs(flat_target), min=1.0e-8)
    metrics = {
        "surrogate_ranking_time_s": float(elapsed),
        "inference_time_per_sample_ms": float((elapsed / max(int(candidates.shape[0]), 1)) * 1000.0),
        "best_of_k_surrogate_mre": float(selected_errors.min().detach().cpu().item()) if int(selected_errors.numel()) else 0.0,
        **metric_payload,
    }
    payload = {
        "ranked_indices": ranked.detach().cpu().numpy().astype(np.int64),
        "selected_indices": selected.detach().cpu().numpy().astype(np.int64),
        "selected_predictions": selected_predictions.detach().cpu().numpy().astype(np.float32),
        "selected_errors": selected_errors.detach().cpu().numpy().astype(np.float32),
        "best_prediction": best_prediction.detach().cpu().numpy().astype(np.float32),
        "best_per_band_mre_pct": (best_rel * 100.0).detach().cpu().numpy().astype(np.float64),
        "best_per_band_mae": best_abs.detach().cpu().numpy().astype(np.float64),
        "best_per_band_rmse": torch.sqrt(torch.square(best_delta)).detach().cpu().numpy().astype(np.float64),
    }
    return payload, metrics


def _candidate_pool_from_record(target_record: Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(target_record.get("candidates", []), dtype=np.float32)
    fallback = np.asarray(target_record.get("raw_candidates", []), dtype=np.float32)
    if raw.ndim == 1 and raw.size:
        raw = raw.reshape(1, -1)
    if fallback.ndim == 1 and fallback.size:
        fallback = fallback.reshape(1, -1)
    if raw.size == 0 and fallback.size:
        raw = fallback
    if raw.size == 0:
        raise ValueError(f"Target {target_record.get('target_id', '<unknown>')} has no candidates to rerank")
    return np.asarray(raw, dtype=np.float32)


def _target_matrix_from_record(
    cfg: GenerativeInverseConfig,
    target_record: Mapping[str, Any],
    *,
    target_shape: tuple[int, int],
) -> np.ndarray:
    condition = _as_mapping(target_record.get("condition"))
    raw = condition.get(cfg.condition_target)
    if raw is None:
        raw = condition.get("target")
    arr = np.asarray(raw, dtype=np.float32)
    if arr.shape == target_shape:
        return arr
    if arr.size == int(np.prod(target_shape)):
        return arr.reshape(target_shape)
    raise ValueError(
        f"Target {target_record.get('target_id', '<unknown>')} condition cannot be reshaped "
        f"to {target_shape}: {arr.shape}"
    )


def _resolve_rerank_candidates_path(cfg: GenerativeInverseConfig) -> Path:
    rerank_cfg = _as_mapping(cfg.rerank)
    configured = rerank_cfg.get("candidates_path") or rerank_cfg.get("source_candidates_path")
    if configured:
        path = Path(str(configured))
        if not path.is_file():
            raise FileNotFoundError(f"rerank.candidates_path not found: {path}")
        return path

    sample_run = _resolve_sample_run(cfg)
    if sample_run is None:
        raise FileNotFoundError("Could not resolve inverse sample candidates; set rerank.candidates_path")
    return sample_run / "candidates.json"


def _strategy_id(strategy: Mapping[str, Any], index: int) -> str:
    explicit = str(strategy.get("id", strategy.get("name", ""))).strip()
    if explicit:
        return explicit
    kind = str(strategy.get("kind", "surrogate")).strip() or "surrogate"
    model = str(strategy.get("model_name", "")).strip()
    input_repr = str(strategy.get("input_repr", "")).strip()
    parts = [kind]
    if model:
        parts.append(model)
    if input_repr:
        parts.append(input_repr)
    parts.append(str(index))
    return "_".join(parts)


def _rerank_strategy_specs(cfg: GenerativeInverseConfig) -> list[dict[str, Any]]:
    rerank_cfg = _as_mapping(cfg.rerank)
    strategies = rerank_cfg.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        base = dict(cfg.surrogate)
        base.setdefault("id", "configured_surrogate")
        strategies = [base]
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(strategies):
        if not isinstance(item, Mapping):
            raise TypeError("rerank.strategies entries must be mappings")
        strategy = dict(item)
        strategy["id"] = _strategy_id(strategy, index)
        resolved.append(strategy)
    return resolved


def _strategy_components(strategy: Mapping[str, Any]) -> list[dict[str, Any]]:
    kind = str(strategy.get("kind", "")).strip().lower()
    if kind == "hybrid":
        raw_components = strategy.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise ValueError(f"Hybrid strategy {strategy.get('id')} requires non-empty components")
        components: list[dict[str, Any]] = []
        for raw_component in raw_components:
            if not isinstance(raw_component, Mapping):
                raise TypeError("Hybrid strategy components must be mappings")
            component = dict(raw_component)
            component.setdefault("weight", 1.0)
            components.append(component)
        return components
    component = dict(strategy)
    component.setdefault("weight", 1.0)
    return [component]


def _component_surrogate_cfg(component: Mapping[str, Any]) -> dict[str, Any]:
    nested = component.get("surrogate")
    if isinstance(nested, Mapping):
        surrogate_cfg = dict(nested)
    else:
        surrogate_cfg = dict(component)
    for key in ("id", "name", "weight", "components", "ranking"):
        surrogate_cfg.pop(key, None)
    return surrogate_cfg


def _error_bundle(
    predictions: Any,
    target: Any,
    *,
    target_shape: tuple[int, int],
    torch: Any,
) -> dict[str, Any]:
    flat_pred = predictions.reshape(int(predictions.shape[0]), -1)
    flat_target = target.reshape(1, -1).expand_as(flat_pred)
    delta = flat_pred - flat_target
    rel = torch.abs(delta) / torch.clamp(torch.abs(flat_target), min=1.0e-8)
    abs_delta = torch.abs(delta)
    squared_delta = torch.square(delta)
    nk, nb = int(target_shape[0]), int(target_shape[1])
    rel_matrix = rel.reshape(int(flat_pred.shape[0]), nk, nb)
    abs_matrix = abs_delta.reshape(int(flat_pred.shape[0]), nk, nb)
    sq_matrix = squared_delta.reshape(int(flat_pred.shape[0]), nk, nb)
    bands_1_n = rel_matrix.reshape(int(flat_pred.shape[0]), -1).mean(dim=1)
    if nb >= 2:
        bands_2_n = rel_matrix[:, :, 1:].reshape(int(flat_pred.shape[0]), -1).mean(dim=1)
    else:
        bands_2_n = bands_1_n
    return {
        "bands_1_n": bands_1_n,
        "bands_2_n": bands_2_n,
        "full": bands_1_n,
        "per_band_mre": rel_matrix.mean(dim=1),
        "per_band_mae": abs_matrix.mean(dim=1),
        "per_band_rmse": torch.sqrt(sq_matrix.mean(dim=1)),
    }


def _build_rerank_strategy(
    cfg: GenerativeInverseConfig,
    strategy: Mapping[str, Any],
    *,
    candidate_dim: int,
    target_shape: tuple[int, int],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    for raw_component in _strategy_components(strategy):
        component_cfg = _component_surrogate_cfg(raw_component)
        ranking_cfg = dict(cfg.ranking)
        if isinstance(strategy.get("ranking"), Mapping):
            ranking_cfg.update(dict(strategy["ranking"]))
        if isinstance(raw_component.get("ranking"), Mapping):
            ranking_cfg.update(dict(raw_component["ranking"]))
        component_inverse_cfg = replace(cfg, surrogate=component_cfg, ranking=ranking_cfg)
        surrogate = _build_surrogate(
            component_inverse_cfg,
            input_dim=candidate_dim,
            target_shape=target_shape,
            torch=torch,
            device=device,
        )
        components.append(
            {
                "weight": float(raw_component.get("weight", 1.0)),
                "cfg": component_inverse_cfg,
                "surrogate": surrogate,
                "surrogate_cfg": component_cfg,
            }
        )
    return {
        "id": str(strategy["id"]),
        "kind": str(strategy.get("kind", "surrogate")),
        "components": components,
        "raw": dict(strategy),
    }


def _rank_with_rerank_strategy(
    *,
    candidates: Any,
    target: Any,
    strategy: Mapping[str, Any],
    target_shape: tuple[int, int],
    top_m: int,
    torch: Any,
    device_info: ResolvedDeviceInfo,
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    device = device_info.torch_device
    synchronize_device(device, torch_module=torch)
    start = time.perf_counter()
    combined_errors = None
    weighted_predictions = None
    total_weight = 0.0
    component_rows: list[dict[str, Any]] = []
    for component in strategy["components"]:
        weight = float(component["weight"])
        if weight <= 0.0:
            continue
        component_cfg = component["cfg"]
        surrogate = component["surrogate"]
        predictions = surrogate.predict_tensor(candidates)
        candidate_errors, metric_payload = _surrogate_error_tensor(predictions, target, component_cfg, torch)
        flat_predictions = predictions.reshape(int(predictions.shape[0]), -1)
        combined_errors = candidate_errors * weight if combined_errors is None else combined_errors + candidate_errors * weight
        weighted_predictions = (
            flat_predictions * weight
            if weighted_predictions is None
            else weighted_predictions + flat_predictions * weight
        )
        total_weight += weight
        component_rows.append(
            {
                "strategy_id": str(strategy["id"]),
                "component_kind": str(component["surrogate_cfg"].get("kind", "")),
                "component_model_name": str(component["surrogate_cfg"].get("model_name", "")),
                "component_input_repr": str(component["surrogate_cfg"].get("input_repr", "")),
                "component_weight": weight,
                "checkpoint_path": str(getattr(surrogate, "checkpoint_path", "")),
                **{key: float(value) for key, value in metric_payload.items() if isinstance(value, (int, float))},
            }
        )
    if combined_errors is None or weighted_predictions is None or total_weight <= 0.0:
        raise ValueError(f"Strategy {strategy['id']} did not produce any positive-weight component scores")

    combined_errors = combined_errors / float(total_weight)
    weighted_predictions = weighted_predictions / float(total_weight)
    ranked = torch.argsort(combined_errors, stable=True)
    selected = ranked[: max(1, min(int(top_m), int(ranked.numel())))]
    synchronize_device(device, torch_module=torch)
    elapsed = time.perf_counter() - start

    selected_predictions = weighted_predictions.index_select(0, selected)
    selected_errors = combined_errors.index_select(0, selected)
    errors = _error_bundle(weighted_predictions, target, target_shape=target_shape, torch=torch)
    selected_bands_1_n = errors["bands_1_n"].index_select(0, selected)
    selected_bands_2_n = errors["bands_2_n"].index_select(0, selected)
    metrics = {
        "surrogate_ranking_time_s": float(elapsed),
        "inference_time_per_sample_ms": float((elapsed / max(int(candidates.shape[0]), 1)) * 1000.0),
        "best_of_k_surrogate_mre": float(selected_errors.min().detach().cpu().item())
        if int(selected_errors.numel())
        else 0.0,
        "best_of_k_surrogate_mre_bands_2_10": float(selected_bands_2_n.min().detach().cpu().item())
        if int(selected_bands_2_n.numel())
        else 0.0,
        "best_of_k_surrogate_mre_bands_1_10": float(selected_bands_1_n.min().detach().cpu().item())
        if int(selected_bands_1_n.numel())
        else 0.0,
        "best_of_k_surrogate_mre_full": float(selected_bands_1_n.min().detach().cpu().item())
        if int(selected_bands_1_n.numel())
        else 0.0,
        "best_of_k_surrogate_mre_bands_2_10_pct": float(selected_bands_2_n.min().detach().cpu().item() * 100.0)
        if int(selected_bands_2_n.numel())
        else 0.0,
        "best_of_k_surrogate_mre_bands_1_10_pct": float(selected_bands_1_n.min().detach().cpu().item() * 100.0)
        if int(selected_bands_1_n.numel())
        else 0.0,
    }
    payload = {
        "ranked_indices": ranked.detach().cpu().numpy().astype(np.int64),
        "selected_indices": selected.detach().cpu().numpy().astype(np.int64),
        "selected_predictions": selected_predictions.reshape(int(selected_predictions.shape[0]), *target_shape)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "selected_errors": selected_errors.detach().cpu().numpy().astype(np.float32),
        "best_prediction": selected_predictions[0].reshape(*target_shape).detach().cpu().numpy().astype(np.float32),
        "best_per_band_mre_pct": (errors["per_band_mre"].index_select(0, selected[:1])[0] * 100.0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "best_per_band_mae": errors["per_band_mae"].index_select(0, selected[:1])[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "best_per_band_rmse": errors["per_band_rmse"].index_select(0, selected[:1])[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
    }
    return payload, metrics, component_rows


def _mean_numeric(rows: list[Mapping[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return float(np.mean(values)) if values else None


def _selection_overlap(a: list[int], b: list[int]) -> float:
    left = set(int(item) for item in a)
    right = set(int(item) for item in b)
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right) / len(union))


def _run_one_target(
    cfg: GenerativeInverseConfig,
    target_index: int,
    target: np.ndarray,
    local_pos: int,
    candidate_dim: int,
    torch: Any,
    device_info: ResolvedDeviceInfo,
    metadata: Mapping[str, Any] | None = None,
    generator_obj: Any | None = None,
    surrogate_obj: Any | None = None,
    secondary_surrogate_obj: Any | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    target_id = _target_id(cfg, target_index, local_pos)
    condition = _target_to_condition_dict(cfg, target_id, target)
    sample_metadata = metadata or {}
    gen_seed = _condition_seed(cfg, target_id, int(target_index))
    device = device_info.torch_device
    target_tensor = torch.as_tensor(np.asarray(target, dtype=np.float32).reshape(-1), dtype=torch.float32, device=device)
    generator = generator_obj or _build_generator(cfg, candidate_dim=candidate_dim, torch=torch, device=device)
    if isinstance(generator, _MockConditionalGenerator):
        generator = _MockConditionalGenerator(
            cfg=cfg,
            dim=candidate_dim,
            target_seed_anchor=float(np.mean(target)),
            seed=gen_seed,
            torch=torch,
            device=device,
        )

    synchronize_device(device, torch_module=torch)
    sample_start = time.perf_counter()
    raw_tensor = generator.sample_tensor(target_tensor, target_id=target_id, target_index=int(target_index))
    synchronize_device(device, torch_module=torch)
    candidate_sampling_time_s = float(time.perf_counter() - sample_start)

    filtered_tensor, filter_metrics, filtered_indices = _filter_candidate_tensor(raw_tensor, cfg, torch)
    raw_batch = generator.to_candidate_batch(raw_tensor, target=target, target_id=target_id, target_index=int(target_index))
    filtered_batch = generator.to_candidate_batch(
        filtered_tensor,
        target=target,
        target_id=target_id,
        target_index=int(target_index),
        metadata_indices=filtered_indices,
    )

    surrogate = surrogate_obj or _build_surrogate(
        cfg,
        input_dim=int(filtered_tensor.reshape(int(filtered_tensor.shape[0]), -1).shape[1]),
        target_shape=tuple(target.shape),
        torch=torch,
        device=device,
    )

    rank_payload, rank_metrics = _rank_candidates_on_device(
        candidates=filtered_tensor,
        surrogate=surrogate,
        target=target_tensor,
        cfg=cfg,
        torch=torch,
        device_info=device_info,
    )
    preflight_metrics: dict[str, float] = {}
    if secondary_surrogate_obj is not None:
        with torch.no_grad():
            primary_predictions = surrogate.predict_tensor(filtered_tensor)
            secondary_predictions = secondary_surrogate_obj.predict_tensor(filtered_tensor)
            preflight_metrics["cross_surrogate_disagreement_bands_2_10"] = float(
                cross_surrogate_consistency_loss(
                    primary_predictions,
                    secondary_predictions,
                    torch=torch,
                    n_bands=_dataset_band_count(str(cfg.dataset_path)),
                    band_start_one_based=2,
                )
                .detach()
                .cpu()
                .item()
            )
            preflight_metrics["gap_descriptor_error"] = float(
                gap_descriptor_loss(
                    primary_predictions,
                    target_tensor,
                    torch=torch,
                    n_bands=_dataset_band_count(str(cfg.dataset_path)),
                    band_start_one_based=2,
                    temperature=float(
                        _as_mapping(cfg.ranking.get("objective")).get("gap_temperature", 0.05)
                    ),
                )
                .detach()
                .cpu()
                .item()
            )

    selected_indices = rank_payload["selected_indices"]
    ranked_indices = rank_payload["ranked_indices"]
    selected = filtered_batch.candidates[selected_indices]
    selected_predictions = rank_payload["selected_predictions"]
    selected_errors = rank_payload["selected_errors"]
    selected_morphology = candidate_morphology_metrics(selected, threshold=cfg.threshold)
    filtered_protocol_metrics = candidate_batch_metrics(filtered_batch, threshold=cfg.threshold)
    metrics = {
        "candidate_sampling_time_s": candidate_sampling_time_s,
        "surrogate_ranking_time_s": float(rank_metrics.get("surrogate_ranking_time_s", 0.0)),
        "inference_time_per_sample_ms": float(rank_metrics.get("inference_time_per_sample_ms", 0.0)),
        "best_of_k_surrogate_mre": float(selected_errors.min()) if selected_errors.size else 0.0,
        "morphology_score": float(selected_morphology.get("mean_morphology_score", 0.0)),
    }
    record = {
        "target_id": target_id,
        "target_index": int(target_index),
        "local_pos": int(local_pos),
        "condition": condition,
        "sample_id": str(sample_metadata.get("sample_id", target_id)),
        "sample_id_source": str(sample_metadata.get("sample_id_source", "missing")),
        "split": str(sample_metadata.get("split", "unassigned")),
        "dataset_id": str(sample_metadata.get("dataset_id", Path(cfg.dataset_path).stem)),
        "candidate_sampling_time_s": candidate_sampling_time_s,
        "raw_candidates": _to_jsonable(raw_batch.candidates),
        "candidates": _to_jsonable(filtered_batch.candidates),
        "metadata": _to_jsonable(filtered_batch.metadata),
        "candidate_seed": int(gen_seed),
        "raw_metrics": candidate_batch_metrics(
            raw_batch,
            threshold=cfg.threshold,
            min_value=cfg.candidate_value_range[0],
            max_value=cfg.candidate_value_range[1],
        ),
        "filtered_metrics": candidate_batch_metrics(
            filtered_batch,
            threshold=cfg.threshold,
            min_value=cfg.candidate_value_range[0],
            max_value=cfg.candidate_value_range[1],
        ),
        "filter_metrics": filter_metrics,
        "surrogate_metrics": {
            **filtered_protocol_metrics,
            **candidate_morphology_metrics(filtered_batch.candidates, threshold=cfg.threshold),
            **rank_metrics,
            **preflight_metrics,
            "binary_diversity_hamming": float(filtered_protocol_metrics.get("diversity_hamming", 0.0)),
            "surrogate_ranking_time_s": float(rank_metrics.get("surrogate_ranking_time_s", 0.0)),
            "best_of_k_surrogate_mre": float(selected_errors.min()) if selected_errors.size else 0.0,
            "selected_morphology": _to_jsonable(selected_morphology),
            "selected_indices": [int(x) for x in selected_indices.tolist()],
            "ranked_indices": [int(x) for x in ranked_indices.tolist()],
            "selected_reconstruction_errors": _to_jsonable(selected_errors),
            "candidate_seed": int(gen_seed),
        },
        "selected_surrogate_predictions": _to_jsonable(selected_predictions),
        "best_surrogate_prediction": _to_jsonable(rank_payload["best_prediction"]),
        "best_per_band_mre_pct": _to_jsonable(rank_payload["best_per_band_mre_pct"]),
        "best_per_band_mae": _to_jsonable(rank_payload["best_per_band_mae"]),
        "best_per_band_rmse": _to_jsonable(rank_payload["best_per_band_rmse"]),
        "selected_indices": [int(x) for x in selected_indices.tolist()],
        "ranked_indices": [int(x) for x in ranked_indices.tolist()],
    }
    return record, metrics


def _run_sample(
    cfg: GenerativeInverseConfig,
    run_dir: Path,
    *,
    torch: Any,
    device_info: ResolvedDeviceInfo,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(cfg.seeds) > 1 and int(cfg.seed) == int(cfg.seeds[0]):
        all_records: list[dict[str, Any]] = []
        seed_metrics: list[dict[str, Any]] = []
        for seed in cfg.seeds:
            seed_cfg = replace(cfg, seed=int(seed), seeds=(int(seed),))
            records, metrics = _run_sample(seed_cfg, run_dir, torch=torch, device_info=device_info)
            all_records.extend(records)
            seed_metrics.append({"seed": int(seed), **metrics})
        numeric_keys = {
            key
            for row in seed_metrics
            for key, value in row.items()
            if key not in {"seed", "targets"} and isinstance(value, (int, float))
        }
        aggregate = {
            "targets": int(seed_metrics[0].get("targets", 0)) if seed_metrics else 0,
            "target_seed_evaluations": int(sum(int(row.get("targets", 0)) for row in seed_metrics)),
            "seeds": [int(row["seed"]) for row in seed_metrics],
            "n_seeds": int(len(seed_metrics)),
        }
        for key in sorted(numeric_keys):
            values = [float(row[key]) for row in seed_metrics if key in row]
            if values:
                aggregate[key] = float(np.mean(values))
        _write_candidate_artifacts(run_dir, all_records)
        return all_records, aggregate

    indices, target_pool, candidate_dim, target_metadata = _load_targets(cfg)
    reset_peak_gpu_memory(device_info.torch_device, torch_module=torch)
    wall_start = time.perf_counter()
    target_records: list[dict[str, Any]] = []
    first_target = np.asarray(target_pool[int(indices[0])], dtype=np.float32).reshape(-1)
    generator_obj = _build_generator(
        cfg,
        candidate_dim=candidate_dim,
        torch=torch,
        device=device_info.torch_device,
    )
    surrogate_obj = _build_surrogate(
        cfg,
        input_dim=candidate_dim,
        target_shape=tuple(first_target.shape),
        torch=torch,
        device=device_info.torch_device,
    )
    objective_cfg = _as_mapping(cfg.ranking.get("objective"))
    secondary_surrogate_obj = _build_objective_secondary_surrogate(
        cfg,
        objective_cfg,
        input_dim=candidate_dim,
        target_shape=tuple(first_target.shape),
        torch=torch,
        device=device_info.torch_device,
    )

    for local_pos, target_index in enumerate(indices):
        target = np.asarray(target_pool[int(target_index)], dtype=np.float32).reshape(-1)
        item, metrics = _run_one_target(
            cfg,
            int(target_index),
            target,
            int(local_pos),
            candidate_dim=candidate_dim,
            torch=torch,
            device_info=device_info,
            metadata=target_metadata[int(local_pos)] if int(local_pos) < len(target_metadata) else None,
            generator_obj=generator_obj,
            surrogate_obj=surrogate_obj,
            secondary_surrogate_obj=secondary_surrogate_obj,
        )
        target_records.append(item)
        del metrics

    _write_candidate_artifacts(run_dir, target_records)

    if target_records:
        validity = [float(item["filtered_metrics"].get("validity_rate", 0.0)) for item in target_records]
        unique = [float(item["filtered_metrics"].get("unique_rate", 0.0)) for item in target_records]
        diversity = [float(item["filtered_metrics"].get("diversity_hamming", 0.0)) for item in target_records]
        nn_min = [float(item["filtered_metrics"].get("nearest_neighbor_hamming_min", 0.0)) for item in target_records]
        nn_p10 = [float(item["filtered_metrics"].get("nearest_neighbor_hamming_p10", 0.0)) for item in target_records]
        nn_mean = [
            float(item["filtered_metrics"].get("nearest_neighbor_hamming_mean", 0.0)) for item in target_records
        ]
        duplicate_clusters = [
            float(item["filtered_metrics"].get("duplicate_cluster_count", 0.0)) for item in target_records
        ]
        largest_duplicate_clusters = [
            float(item["filtered_metrics"].get("largest_duplicate_cluster_fraction", 0.0))
            for item in target_records
        ]
        per_target_duplicate_diagnostics = [
            {
                "target_id": str(item["target_id"]),
                "target_index": int(item["target_index"]),
                "unique_rate": float(item["filtered_metrics"].get("unique_rate", 0.0)),
                "diversity_hamming": float(item["filtered_metrics"].get("diversity_hamming", 0.0)),
                "nearest_neighbor_hamming_min": float(
                    item["filtered_metrics"].get("nearest_neighbor_hamming_min", 0.0)
                ),
                "nearest_neighbor_hamming_p10": float(
                    item["filtered_metrics"].get("nearest_neighbor_hamming_p10", 0.0)
                ),
                "nearest_neighbor_hamming_mean": float(
                    item["filtered_metrics"].get("nearest_neighbor_hamming_mean", 0.0)
                ),
                "duplicate_cluster_count": float(item["filtered_metrics"].get("duplicate_cluster_count", 0.0)),
                "largest_duplicate_cluster_fraction": float(
                    item["filtered_metrics"].get("largest_duplicate_cluster_fraction", 0.0)
                ),
            }
            for item in target_records
        ]
        binary_diversity = [
            float(item["surrogate_metrics"].get("binary_diversity_hamming", 0.0)) for item in target_records
        ]
        morphology = [float(item["surrogate_metrics"].get("selected_morphology", {}).get("mean_morphology_score", 0.0)) for item in target_records]
        best = [float(item["surrogate_metrics"].get("best_of_k_surrogate_mre", 0.0)) for item in target_records]
        best_bands = [
            float(item["surrogate_metrics"].get("best_of_k_surrogate_mre_bands_2_10", 0.0))
            for item in target_records
            if item["surrogate_metrics"].get("best_of_k_surrogate_mre_bands_2_10") is not None
        ]
        best_full = [
            float(item["surrogate_metrics"].get("best_of_k_surrogate_mre_full", 0.0))
            for item in target_records
            if item["surrogate_metrics"].get("best_of_k_surrogate_mre_full") is not None
        ]
        candidate_time = [float(item["candidate_sampling_time_s"]) for item in target_records]
        rank_time = [float(item["surrogate_metrics"].get("surrogate_ranking_time_s", 0.0)) for item in target_records]
        infer_time = [float(item["surrogate_metrics"].get("inference_time_per_sample_ms", 0.0)) for item in target_records]
        filter_ratio = [float(item["filter_metrics"].get("filter_pass_ratio", 0.0)) for item in target_records]
        cross_disagreement = [
            float(item["surrogate_metrics"].get("cross_surrogate_disagreement_bands_2_10", 0.0))
            for item in target_records
            if item["surrogate_metrics"].get("cross_surrogate_disagreement_bands_2_10") is not None
        ]
        gap_descriptor = [
            float(item["surrogate_metrics"].get("gap_descriptor_error", 0.0))
            for item in target_records
            if item["surrogate_metrics"].get("gap_descriptor_error") is not None
        ]
    else:
        validity = unique = diversity = nn_min = nn_p10 = nn_mean = duplicate_clusters = largest_duplicate_clusters = []
        binary_diversity = morphology = best = best_bands = best_full = candidate_time = rank_time = infer_time = []
        filter_ratio = cross_disagreement = gap_descriptor = []
        per_target_duplicate_diagnostics = []

    metrics = {
        "targets": int(len(target_records)),
        "candidate_dim": int(candidate_dim),
        "candidate_sampling_time_s": float(sum(candidate_time)),
        "surrogate_ranking_time_s": float(sum(rank_time)),
        "inference_time_per_sample_ms": float(np.mean(infer_time) if infer_time else 0.0),
        "candidate_sampling_time_per_target_ms": float(1000.0 * np.mean(candidate_time) if candidate_time else 0.0),
        "validity_rate": float(np.mean(validity) if validity else 0.0),
        "unique_rate": float(np.mean(unique) if unique else 0.0),
        "diversity_hamming": float(np.mean(diversity) if diversity else 0.0),
        "nearest_neighbor_hamming_min": float(np.min(nn_min) if nn_min else 0.0),
        "nearest_neighbor_hamming_p10": float(np.mean(nn_p10) if nn_p10 else 0.0),
        "nearest_neighbor_hamming_mean": float(np.mean(nn_mean) if nn_mean else 0.0),
        "duplicate_cluster_count": float(np.mean(duplicate_clusters) if duplicate_clusters else 0.0),
        "largest_duplicate_cluster_fraction": float(
            np.mean(largest_duplicate_clusters) if largest_duplicate_clusters else 0.0
        ),
        "binary_diversity_hamming": float(np.mean(binary_diversity) if binary_diversity else 0.0),
        "morphology_score": float(np.mean(morphology) if morphology else 0.0),
        "best_of_k_surrogate_mre": float(np.mean(best) if best else 0.0),
        "best_of_k_surrogate_mre_bands_2_10": float(np.mean(best_bands) if best_bands else 0.0),
        "best_of_k_surrogate_mre_full": float(np.mean(best_full) if best_full else 0.0),
        "cross_surrogate_disagreement_bands_2_10": float(np.mean(cross_disagreement) if cross_disagreement else 0.0),
        "gap_descriptor_error": float(np.mean(gap_descriptor) if gap_descriptor else 0.0),
        "filter_pass_ratio": float(np.mean(filter_ratio) if filter_ratio else 0.0),
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "batch_size": int(cfg.n_candidates),
        "effective_batch_size": int(cfg.n_candidates),
        "hard_duplicate_diagnostics": {
            "global_mean": {
                "unique_rate": float(np.mean(unique) if unique else 0.0),
                "diversity_hamming": float(np.mean(diversity) if diversity else 0.0),
                "nearest_neighbor_hamming_min": float(np.min(nn_min) if nn_min else 0.0),
                "nearest_neighbor_hamming_p10": float(np.mean(nn_p10) if nn_p10 else 0.0),
                "nearest_neighbor_hamming_mean": float(np.mean(nn_mean) if nn_mean else 0.0),
                "duplicate_cluster_count": float(np.mean(duplicate_clusters) if duplicate_clusters else 0.0),
                "largest_duplicate_cluster_fraction": float(
                    np.mean(largest_duplicate_clusters) if largest_duplicate_clusters else 0.0
                ),
            },
            "per_target": per_target_duplicate_diagnostics,
        },
        **_device_metadata_with_peak(device_info, torch),
    }
    return target_records, metrics


def _resolve_sample_run(cfg: GenerativeInverseConfig) -> Path | None:
    if cfg.run_dir is not None and (cfg.run_dir / "candidates.json").exists():
        return cfg.run_dir

    runs_root = cfg.runs_dir
    if not runs_root.exists():
        return None

    candidates: list[Path] = []
    for path in runs_root.rglob("candidates.json"):
        if path.parent.name.startswith(cfg.experiment_id):
            candidates.append(path.parent)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _write_candidate_artifacts(run_dir: Path, target_records: list[dict[str, Any]]) -> None:
    (run_dir / "candidates.json").write_text(
        json.dumps({"version": 1, "targets": target_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_history(
        run_dir / "ranked_candidates.csv",
        [
            {
                "target_id": item["target_id"],
                "target_index": int(item["target_index"]),
                "sample_id": item.get("sample_id", ""),
                "candidate_count_filtered": float(item.get("filtered_metrics", {}).get("n_candidates", 0.0)),
                "unique_rate": float(item.get("filtered_metrics", {}).get("unique_rate", 0.0)),
                "diversity_hamming": float(item.get("filtered_metrics", {}).get("diversity_hamming", 0.0)),
                "selected_indices": json.dumps(item.get("selected_indices", []), ensure_ascii=False),
                "ranked_indices": json.dumps(item.get("ranked_indices", []), ensure_ascii=False),
                "best_of_k_surrogate_mre": float(
                    item.get("surrogate_metrics", {}).get("best_of_k_surrogate_mre", 0.0)
                ),
                "best_of_k_surrogate_mre_bands_2_10": float(
                    item.get("surrogate_metrics", {}).get("best_of_k_surrogate_mre_bands_2_10", 0.0)
                ),
                "best_of_k_surrogate_mre_full": float(
                    item.get("surrogate_metrics", {}).get("best_of_k_surrogate_mre_full", 0.0)
                ),
                "nearest_neighbor_hamming_min": float(
                    item.get("filtered_metrics", {}).get("nearest_neighbor_hamming_min", 0.0)
                ),
                "nearest_neighbor_hamming_p10": float(
                    item.get("filtered_metrics", {}).get("nearest_neighbor_hamming_p10", 0.0)
                ),
                "nearest_neighbor_hamming_mean": float(
                    item.get("filtered_metrics", {}).get("nearest_neighbor_hamming_mean", 0.0)
                ),
                "duplicate_cluster_count": float(item.get("filtered_metrics", {}).get("duplicate_cluster_count", 0.0)),
                "largest_duplicate_cluster_fraction": float(
                    item.get("filtered_metrics", {}).get("largest_duplicate_cluster_fraction", 0.0)
                ),
                "cross_surrogate_disagreement_bands_2_10": float(
                    item.get("surrogate_metrics", {}).get("cross_surrogate_disagreement_bands_2_10", 0.0)
                ),
                "binary_diversity_hamming": float(
                    item.get("surrogate_metrics", {}).get("binary_diversity_hamming", 0.0)
                ),
                "gap_descriptor_error": float(
                    item.get("surrogate_metrics", {}).get("gap_descriptor_error", 0.0)
                ),
                "surrogate_ranking_time_s": float(
                    item.get("surrogate_metrics", {}).get("surrogate_ranking_time_s", 0.0)
                ),
            }
            for item in target_records
        ],
    )


def run_inverse_rerank(config_path: str | Path) -> Path:
    cfg = load_inverse_run_config(config_path)
    torch = _require_torch()
    device_info = _resolve_runtime_device(cfg, torch)
    rerank_cfg = _as_mapping(cfg.rerank)
    source_candidates_path = _resolve_rerank_candidates_path(cfg)
    run_dir = make_run_dir(cfg.experiment_id, runs_dir=cfg.runs_dir, run_dir=cfg.run_dir)
    run_start = datetime.now().astimezone()
    run_meta = {
        "script": "phc2d inverse rerank",
        "config_path": str(cfg.source_path),
        "dataset_path": str(cfg.dataset_path),
        "experiment_id": cfg.experiment_id,
        "validation_boundary": "inverse_surrogate_rerank_no_mpb_meep",
        "source_candidates_path": str(source_candidates_path),
        "pid": os.getpid(),
        "start_timestamp": run_start.isoformat(),
        **collect_device_metadata(device_info, cfg.runtime),
    }
    logger = setup_logging(log_file=run_dir / "run.log", run_meta=run_meta)
    write_resolved_config(run_dir, cfg.to_resolved_yaml())
    ensure_figures_dir(run_dir)
    reset_peak_gpu_memory(device_info.torch_device, torch_module=torch)

    candidate_text = source_candidates_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(candidate_text)
    except json.JSONDecodeError:
        payload = yaml.safe_load(candidate_text)
    targets = payload.get("targets", []) if isinstance(payload, Mapping) else []
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"No target records found in {source_candidates_path}")

    requested_targets = rerank_cfg.get("n_targets", rerank_cfg.get("target_limit"))
    if requested_targets is not None:
        targets = targets[: max(1, min(int(requested_targets), len(targets)))]

    bundle = load_forward_dataset(cfg.dataset_path)
    target_shape = (int(bundle.nk), int(bundle.nb))
    first_candidates = _candidate_pool_from_record(targets[0])
    candidate_dim = int(first_candidates.reshape(int(first_candidates.shape[0]), -1).shape[1])
    top_m = int(rerank_cfg.get("top_m", cfg.surrogate_top_m))
    strategy_specs = _rerank_strategy_specs(cfg)
    strategy_runtimes = [
        _build_rerank_strategy(
            cfg,
            strategy,
            candidate_dim=candidate_dim,
            target_shape=target_shape,
            torch=torch,
            device=device_info.torch_device,
        )
        for strategy in strategy_specs
    ]
    if not strategy_runtimes:
        raise ValueError("At least one rerank strategy is required")
    primary_strategy_id = str(rerank_cfg.get("primary_strategy", strategy_runtimes[0]["id"]))

    wall_start = time.perf_counter()
    target_records: list[dict[str, Any]] = []
    per_target_rows: list[dict[str, Any]] = []
    per_band_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for local_pos, raw_target in enumerate(targets):
        target_record = dict(raw_target)
        target_id = str(target_record.get("target_id", f"target_{local_pos:06d}"))
        target_index = int(target_record.get("target_index", local_pos))
        candidates_np = _candidate_pool_from_record(target_record)
        target_matrix = _target_matrix_from_record(cfg, target_record, target_shape=target_shape)
        candidates_tensor = torch.as_tensor(candidates_np, dtype=torch.float32, device=device_info.torch_device)
        target_tensor = torch.as_tensor(target_matrix.reshape(-1), dtype=torch.float32, device=device_info.torch_device)
        strategy_rankings: dict[str, Any] = {}
        target_strategy_rows: list[dict[str, Any]] = []

        for strategy in strategy_runtimes:
            strategy_id = str(strategy["id"])
            rank_payload, rank_metrics, rows = _rank_with_rerank_strategy(
                candidates=candidates_tensor,
                target=target_tensor,
                strategy=strategy,
                target_shape=target_shape,
                top_m=top_m,
                torch=torch,
                device_info=device_info,
            )
            selected_indices = [int(item) for item in rank_payload["selected_indices"].tolist()]
            ranked_indices = [int(item) for item in rank_payload["ranked_indices"].tolist()]
            strategy_rankings[strategy_id] = {
                "strategy_id": strategy_id,
                "strategy_kind": str(strategy["kind"]),
                "selected_indices": selected_indices,
                "ranked_indices": ranked_indices,
                "selected_errors": rank_payload["selected_errors"],
                "selected_surrogate_predictions": rank_payload["selected_predictions"],
                "surrogate_prediction_source": "weighted_strategy_surrogate_prediction",
                "surrogate_metrics": rank_metrics,
                "components": [
                    {
                        "kind": str(component["surrogate_cfg"].get("kind", "")),
                        "model_name": str(component["surrogate_cfg"].get("model_name", "")),
                        "input_repr": str(component["surrogate_cfg"].get("input_repr", "")),
                        "weight": float(component["weight"]),
                        "checkpoint_path": str(getattr(component["surrogate"], "checkpoint_path", "")),
                    }
                    for component in strategy["components"]
                ],
            }
            row = {
                "target_id": target_id,
                "target_index": target_index,
                "local_pos": int(local_pos),
                "strategy_id": strategy_id,
                "strategy_kind": str(strategy["kind"]),
                "candidate_count": int(candidates_np.shape[0]),
                "selected_indices": json.dumps(selected_indices, ensure_ascii=False),
                "ranked_indices": json.dumps(ranked_indices, ensure_ascii=False),
                **rank_metrics,
            }
            target_strategy_rows.append(row)
            for component_row in rows:
                component = dict(component_row)
                component["target_id"] = target_id
                component["target_index"] = target_index
                component_rows.append(component)
            for band_idx in range(int(bundle.nb)):
                per_band_rows.append(
                    {
                        "target_id": target_id,
                        "target_index": target_index,
                        "strategy_id": strategy_id,
                        "band_one_based": int(band_idx + 1),
                        "surrogate_mre_pct": float(rank_payload["best_per_band_mre_pct"][band_idx]),
                        "mae": float(rank_payload["best_per_band_mae"][band_idx]),
                        "rmse": float(rank_payload["best_per_band_rmse"][band_idx]),
                        "sample_count": int(bundle.nk),
                    }
                )

        baseline_indices: list[int] = []
        if primary_strategy_id in strategy_rankings:
            baseline_indices = [int(item) for item in strategy_rankings[primary_strategy_id]["selected_indices"]]
        elif strategy_rankings:
            first_ranking = next(iter(strategy_rankings.values()))
            baseline_indices = [int(item) for item in first_ranking.get("selected_indices", [])]
        for row in target_strategy_rows:
            selected = json.loads(str(row["selected_indices"]))
            row["selection_overlap_with_primary"] = _selection_overlap(baseline_indices, [int(item) for item in selected])
            per_target_rows.append(row)

        primary = strategy_rankings.get(primary_strategy_id, next(iter(strategy_rankings.values())))
        target_record["strategy_rankings"] = strategy_rankings
        target_record["selected_indices"] = primary["selected_indices"]
        target_record["ranked_indices"] = primary["ranked_indices"]
        target_record["selected_surrogate_predictions"] = primary["selected_surrogate_predictions"]
        target_record["surrogate_metrics"] = dict(primary["surrogate_metrics"])
        target_record["surrogate_metrics"]["strategy_id"] = str(primary.get("strategy_id", primary_strategy_id))
        target_records.append(target_record)

    strategy_rows: list[dict[str, Any]] = []
    for strategy in strategy_runtimes:
        strategy_id = str(strategy["id"])
        rows = [row for row in per_target_rows if row["strategy_id"] == strategy_id]
        selected_sets = [json.loads(str(row["selected_indices"])) for row in rows]
        unique_selected = {
            (str(row["target_id"]), int(index))
            for row, selected in zip(rows, selected_sets)
            for index in selected
        }
        strategy_rows.append(
            {
                "strategy_id": strategy_id,
                "strategy_kind": str(strategy["kind"]),
                "target_count": int(len(rows)),
                "selected_candidate_count": int(sum(len(selected) for selected in selected_sets)),
                "unique_target_candidate_count": int(len(unique_selected)),
                "best_of_k_surrogate_mre_bands_2_10_pct": _mean_numeric(
                    rows, "best_of_k_surrogate_mre_bands_2_10_pct"
                ),
                "best_of_k_surrogate_mre_bands_1_10_pct": _mean_numeric(
                    rows, "best_of_k_surrogate_mre_bands_1_10_pct"
                ),
                "best_of_k_surrogate_mre_full": _mean_numeric(rows, "best_of_k_surrogate_mre_full"),
                "surrogate_ranking_time_s": _mean_numeric(rows, "surrogate_ranking_time_s"),
                "inference_time_per_sample_ms": _mean_numeric(rows, "inference_time_per_sample_ms"),
                "selection_overlap_with_primary": _mean_numeric(rows, "selection_overlap_with_primary"),
            }
        )

    aggregate_band_rows: list[dict[str, Any]] = []
    for strategy in strategy_runtimes:
        strategy_id = str(strategy["id"])
        rows = [row for row in per_band_rows if row["strategy_id"] == strategy_id]
        for band_idx in range(int(bundle.nb)):
            band_rows = [row for row in rows if int(row["band_one_based"]) == band_idx + 1]
            aggregate_band_rows.append(
                {
                    "strategy_id": strategy_id,
                    "band_one_based": int(band_idx + 1),
                    "surrogate_mre_pct": _mean_numeric(band_rows, "surrogate_mre_pct"),
                    "mae": _mean_numeric(band_rows, "mae"),
                    "rmse": _mean_numeric(band_rows, "rmse"),
                    "sample_count": int(sum(int(row["sample_count"]) for row in band_rows)),
                }
            )

    strategy_payload = {
        "version": 2,
        "kind": "inverse_surrogate_rerank_candidates",
        "source_candidates_path": str(source_candidates_path),
        "dataset_path": str(cfg.dataset_path),
        "validation_boundary": "inverse_surrogate_rerank_no_mpb_meep",
        "target_shape": list(target_shape),
        "top_m": int(top_m),
        "primary_strategy": primary_strategy_id,
        "strategies": [
            {
                "strategy_id": str(strategy["id"]),
                "strategy_kind": str(strategy["kind"]),
                "components": [
                    {
                        "kind": str(component["surrogate_cfg"].get("kind", "")),
                        "model_name": str(component["surrogate_cfg"].get("model_name", "")),
                        "input_repr": str(component["surrogate_cfg"].get("input_repr", "")),
                        "weight": float(component["weight"]),
                        "checkpoint_path": str(getattr(component["surrogate"], "checkpoint_path", "")),
                    }
                    for component in strategy["components"]
                ],
            }
            for strategy in strategy_runtimes
        ],
        "targets": target_records,
    }
    (run_dir / "strategy_candidates.json").write_text(
        json.dumps(_to_jsonable(strategy_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_history(run_dir / "rerank_leaderboard.csv", strategy_rows)
    _write_history(run_dir / "per_strategy_metrics.csv", strategy_rows)
    _write_history(run_dir / "per_target_strategy_metrics.csv", per_target_rows)
    _write_history(run_dir / "per_band_surrogate_metrics.csv", aggregate_band_rows)
    _write_history(run_dir / "per_target_band_surrogate_metrics.csv", per_band_rows)
    _write_history(run_dir / "component_metrics.csv", component_rows)
    _write_history(
        run_dir / "history.csv",
        [
            {
                "stage": "rerank",
                "targets": int(len(target_records)),
                "strategy_count": int(len(strategy_runtimes)),
                "top_m": int(top_m),
                "wall_time_total_s": float(time.perf_counter() - wall_start),
            }
        ],
    )

    primary_rows = [row for row in strategy_rows if row["strategy_id"] == primary_strategy_id]
    primary_metrics = dict(primary_rows[0]) if primary_rows else dict(strategy_rows[0])
    metrics_payload = {
        "status": "ok",
        "stage": "inverse_surrogate_rerank",
        "validation_boundary": "inverse_surrogate_rerank_no_mpb_meep",
        "targets": int(len(target_records)),
        "strategy_count": int(len(strategy_runtimes)),
        "top_m": int(top_m),
        "primary_strategy": primary_strategy_id,
        "strategy_candidates_path": "strategy_candidates.json",
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        **{f"primary_{key}": value for key, value in primary_metrics.items() if key not in {"strategy_id", "strategy_kind"}},
        **_device_metadata_with_peak(device_info, torch),
    }
    write_metrics(run_dir, metrics_payload)
    write_results(
        run_dir,
        {
            "action": "inverse rerank",
            "stage": "inverse_surrogate_rerank",
            "run_dir": str(run_dir),
            "experiment_id": cfg.experiment_id,
            "dataset_path": str(cfg.dataset_path),
            "source_candidates_path": str(source_candidates_path),
            "strategy_candidates_path": "strategy_candidates.json",
            "rerank_leaderboard_path": "rerank_leaderboard.csv",
            "per_strategy_metrics_path": "per_strategy_metrics.csv",
            "per_target_strategy_metrics_path": "per_target_strategy_metrics.csv",
            "per_band_surrogate_metrics_path": "per_band_surrogate_metrics.csv",
            "validation_boundary": "inverse_surrogate_rerank_no_mpb_meep",
            "surrogate_side_no_mpb_meep": True,
        },
    )
    try:
        plot_metric_report(metrics_payload, run_dir / "figures" / "metrics.png", title="inverse rerank metrics")
    except Exception:
        logger.warning("Could not write inverse rerank metric figure.")
    run_meta.update(metrics_payload)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    logger.info("Inverse rerank finished: %s", run_dir)
    return run_dir


def _train_one_generator(
    cfg: GenerativeInverseConfig,
    *,
    seed: int,
    run_dir: Path,
    torch: Any,
    device_info: ResolvedDeviceInfo,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    bundle = load_forward_dataset(cfg.dataset_path)
    train_indices = np.asarray(bundle.idx_train, dtype=np.int64)
    if train_indices.size == 0:
        raise ValueError(f"No train split available in {cfg.dataset_path}")
    condition = np.asarray(bundle.Wh[train_indices], dtype=np.float32).reshape(int(train_indices.size), -1)
    target = np.asarray(bundle.rho[train_indices], dtype=np.float32).reshape(int(train_indices.size), -1)
    device = device_info.torch_device
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    x = torch.as_tensor(condition, dtype=torch.float32, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    net = _make_generator_network(
        cfg,
        condition_dim=int(x.shape[1]),
        candidate_dim=int(y.shape[1]),
        torch=torch,
    ).to(device)
    optimizer = _make_inverse_optimizer(cfg, net.parameters(), torch=torch)
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator(device=str(device))
    generator.manual_seed(int(seed))
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    n = int(x.shape[0])
    batch_size = max(1, int(cfg.train_batch_size))
    suffix = _seed_lr_suffix(cfg, seed)
    for epoch in range(int(cfg.train_epochs)):
        order = torch.randperm(n, generator=generator, device=device)
        losses: list[float] = []
        net.train()
        for batch_start in range(0, n, batch_size):
            idx = order[batch_start : batch_start + batch_size]
            xb = x.index_select(0, idx)
            yb = y.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True)
            pred = net.reconstruct(xb, torch)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append(
            {
                "stage": "train",
                "seed": int(seed),
                "learning_rate": float(cfg.learning_rate),
                "epoch": int(epoch + 1),
                "train_loss": float(np.mean(losses) if losses else 0.0),
            }
        )
        if int(epoch + 1) in set(cfg.checkpoint_epochs):
            torch.save(
                {
                    "model_name": cfg.model_name,
                    "generator_kind": _generator_kind(cfg),
                    "seed": int(seed),
                    "condition_dim": int(x.shape[1]),
                    "candidate_dim": int(y.shape[1]),
                    "latent_dim": int(cfg.latent_dim),
                    "hidden_dim": int(cfg.hidden_dim),
                    "depth": int(cfg.depth),
                    "mode_count": int(cfg.mode_count),
                    "learning_rate": float(cfg.learning_rate),
                    "optimizer": cfg.optimizer,
                    "weight_decay": float(cfg.weight_decay),
                    "state_dict": net.state_dict(),
                    "dataset_path": str(cfg.dataset_path),
                    "condition_target": cfg.condition_target,
                    "output_repr": cfg.output_repr,
                    "epoch": int(epoch + 1),
                },
                run_dir / f"checkpoint_epoch_{int(epoch + 1)}_{suffix}.pt",
            )
    synchronize_device(device, torch_module=torch)
    train_time_s = float(time.perf_counter() - start)
    checkpoint = run_dir / f"model_{suffix}.pt"
    torch.save(
        {
            "model_name": cfg.model_name,
            "generator_kind": _generator_kind(cfg),
            "seed": int(seed),
            "condition_dim": int(x.shape[1]),
            "candidate_dim": int(y.shape[1]),
            "latent_dim": int(cfg.latent_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "depth": int(cfg.depth),
            "mode_count": int(cfg.mode_count),
            "learning_rate": float(cfg.learning_rate),
            "optimizer": cfg.optimizer,
            "weight_decay": float(cfg.weight_decay),
            "state_dict": net.state_dict(),
            "dataset_path": str(cfg.dataset_path),
            "condition_target": cfg.condition_target,
            "output_repr": cfg.output_repr,
        },
        checkpoint,
    )
    metrics = {
        "seed": int(seed),
        "learning_rate": float(cfg.learning_rate),
        "train_time_s": train_time_s,
        "final_train_loss": float(history[-1]["train_loss"] if history else 0.0),
        "checkpoint_path": str(checkpoint),
    }
    return checkpoint, history, metrics


def run_inverse_train(config_path: str | Path) -> Path:
    cfg = load_inverse_run_config(config_path)
    torch = _require_torch()
    device_info = _resolve_runtime_device(cfg, torch)
    run_dir = make_run_dir(cfg.experiment_id, runs_dir=cfg.runs_dir, run_dir=cfg.run_dir)
    run_start = datetime.now().astimezone()
    run_meta = {
        "script": "phc2d inverse train",
        "config_path": str(cfg.source_path),
        "dataset_path": str(cfg.dataset_path),
        "experiment_id": cfg.experiment_id,
        "validation_boundary": "inverse_generator_train_surrogate_side_no_mpb_meep",
        "pid": os.getpid(),
        "start_timestamp": run_start.isoformat(),
        **collect_device_metadata(device_info, cfg.runtime),
    }
    logger = setup_logging(log_file=run_dir / "run.log", run_meta=run_meta)
    write_resolved_config(run_dir, cfg.to_resolved_yaml())
    ensure_figures_dir(run_dir)
    reset_peak_gpu_memory(device_info.torch_device, torch_module=torch)

    wall_start = time.perf_counter()
    history: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    learning_rates = _configured_learning_rates(cfg)
    for learning_rate in learning_rates:
        for seed in cfg.seeds:
            checkpoint, rows, metrics = _train_one_generator(
                replace(cfg, seed=int(seed), learning_rate=float(learning_rate), seeds=(int(seed),)),
                seed=int(seed),
                run_dir=run_dir,
                torch=torch,
                device_info=device_info,
            )
            history.extend(rows)
            seed_rows.append(metrics)
            checkpoints.append(str(checkpoint))
    if checkpoints:
        latest_payload = torch.load(checkpoints[0], map_location=device_info.torch_device, weights_only=False)
        torch.save(latest_payload, run_dir / "model.pt")

    train_time_s = float(sum(float(row["train_time_s"]) for row in seed_rows))
    metrics_payload = {
        "status": "ok",
        "stage": "inverse_train",
        "seeds": [int(seed) for seed in cfg.seeds],
        "n_seeds": int(len(cfg.seeds)),
        "learning_rates": [float(item) for item in learning_rates],
        "n_learning_rates": int(len(learning_rates)),
        "train_time_s": train_time_s,
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "final_train_loss_mean": float(np.mean([float(row["final_train_loss"]) for row in seed_rows]))
        if seed_rows
        else 0.0,
        "checkpoint_paths": checkpoints,
        **_device_metadata_with_peak(device_info, torch),
    }
    _write_history(run_dir / "history.csv", history)
    _write_history(run_dir / "seed_metrics.csv", seed_rows)
    write_metrics(run_dir, metrics_payload)
    write_results(
        run_dir,
        {
            "action": "inverse train",
            "stage": "inverse_train",
            "run_dir": str(run_dir),
            "experiment_id": cfg.experiment_id,
            "dataset_path": str(cfg.dataset_path),
            "checkpoint_paths": checkpoints,
            "learning_rates": [float(item) for item in learning_rates],
            "surrogate_side_no_mpb_meep": True,
        },
    )
    logger.info("Inverse train finished: %s", run_dir)
    run_meta.update(metrics_payload)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    return run_dir


def _surrogate_predict_for_objective(surrogate: Any, candidates: Any) -> Any:
    predictions = surrogate.predict_tensor(candidates)
    if not bool(getattr(predictions, "requires_grad", False)):
        raise ValueError(
            "Objective-driven inverse training requires a differentiable surrogate path. "
            "Use a mock differentiable surrogate for smoke tests or a rho_wedge forward surrogate "
            "that preserves gradients."
        )
    return predictions


def _objective_training_candidates(
    net: Any,
    xb: Any,
    yb: Any,
    *,
    objective: Mapping[str, Any],
    seed: int,
    epoch: int,
    batch_start: int,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, Any, int]:
    group_size = max(1, int(objective.get("n_objective_candidates", objective.get("candidate_group_size", 1)) or 1))
    if group_size <= 1 or int(getattr(net, "latent_dim", 0)) <= 0:
        return net.reconstruct(xb, torch), xb, yb, 1

    generator = torch.Generator(device=str(device))
    generator.manual_seed(_stable_seed(seed, "objective_train", epoch, batch_start))
    condition = xb.repeat_interleave(group_size, dim=0)
    reference = yb.repeat_interleave(group_size, dim=0)
    noise = torch.randn(
        (int(condition.shape[0]), int(net.latent_dim)),
        generator=generator,
        dtype=xb.dtype,
        device=device,
    )
    if hasattr(net, "module"):
        candidates = net.module(torch.cat([condition, noise], dim=1))
    elif hasattr(net, "_decode"):
        mode_count = max(1, int(getattr(net, "mode_count", 1)))
        per_condition_modes = torch.arange(group_size, device=device, dtype=torch.long) % mode_count
        mode_ids = per_condition_modes.repeat(int(xb.shape[0]))
        candidates = net._decode(condition, noise, mode_ids, torch)
    else:
        return net.reconstruct(xb, torch), xb, yb, 1
    return candidates, condition, reference, group_size


def _latent_input_gradient_norm(net: Any, *, torch: Any) -> Any:
    """Return the gradient norm of decoder columns connected to latent noise."""

    latent_dim = max(0, int(getattr(net, "latent_dim", 0)))
    if latent_dim <= 0:
        return torch.zeros((), dtype=torch.float32)
    first_layer = None
    if hasattr(net, "decoder"):
        first_layer = net.decoder[0]
    elif hasattr(net, "module"):
        first_layer = net.module[0]
    gradient = None if first_layer is None else getattr(first_layer.weight, "grad", None)
    if gradient is None:
        parameter = next(iter(net.parameters()))
        return parameter.new_zeros(())
    return torch.linalg.vector_norm(gradient[:, -latent_dim:])


def _latest_objective_checkpoint(run_dir: Path, *, suffix: str) -> tuple[Path | None, int]:
    candidates: list[tuple[int, Path]] = []
    prefix = "objective_checkpoint_epoch_"
    marker = f"_{suffix}.pt"
    for path in run_dir.glob(f"{prefix}*{marker}"):
        name = path.name
        try:
            epoch = int(name[len(prefix) : -len(marker)])
        except ValueError:
            continue
        candidates.append((epoch, path))
    if not candidates:
        return None, 0
    epoch, path = max(candidates, key=lambda item: item[0])
    return path, int(epoch)


def _train_one_objective_generator(
    cfg: GenerativeInverseConfig,
    *,
    seed: int,
    run_dir: Path,
    torch: Any,
    device_info: ResolvedDeviceInfo,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    bundle = load_forward_dataset(cfg.dataset_path)
    train_indices = np.asarray(bundle.idx_train, dtype=np.int64)
    if train_indices.size == 0:
        raise ValueError(f"No train split available in {cfg.dataset_path}")
    condition = np.asarray(bundle.Wh[train_indices], dtype=np.float32).reshape(int(train_indices.size), -1)
    target_rho = np.asarray(bundle.rho[train_indices], dtype=np.float32).reshape(int(train_indices.size), -1)
    device = device_info.torch_device
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    x = torch.as_tensor(condition, dtype=torch.float32, device=device)
    y = torch.as_tensor(target_rho, dtype=torch.float32, device=device)
    net = _make_generator_network(
        cfg,
        condition_dim=int(x.shape[1]),
        candidate_dim=int(y.shape[1]),
        torch=torch,
    ).to(device)

    objective_cfg = _as_mapping(cfg.ranking.get("objective"))
    if not objective_cfg:
        objective_cfg = {
            "surrogate_weight": float(cfg.ranking.get("surrogate_weight", 1.0)),
            "reconstruction_weight": float(cfg.ranking.get("reconstruction_weight", 0.05)),
            "band_start_one_based": float(cfg.ranking.get("band_start_one_based", 2)),
        }
    suffix = (
        f"seed{int(seed)}_{_learning_rate_label(cfg.learning_rate)}"
        if bool(objective_cfg.get("checkpoint_name_include_lr", False))
        else _seed_lr_suffix(cfg, seed)
    )
    resume_checkpoint, resume_epoch = _latest_objective_checkpoint(run_dir, suffix=suffix)
    checkpoint = resume_checkpoint or _find_latest_generator_checkpoint(cfg)
    checkpoint_payload = None
    if checkpoint is not None and checkpoint.is_file():
        checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint_payload["state_dict"])

    surrogate = _build_surrogate(
        cfg,
        input_dim=int(y.shape[1]),
        target_shape=(int(x.shape[1]),),
        torch=torch,
        device=device,
    )
    optimizer = _make_inverse_optimizer(cfg, net.parameters(), torch=torch)
    if resume_checkpoint is not None and checkpoint_payload is not None:
        optimizer_state = checkpoint_payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    secondary_surrogate = _build_objective_secondary_surrogate(
        cfg,
        objective_cfg,
        input_dim=int(y.shape[1]),
        target_shape=(int(x.shape[1]),),
        torch=torch,
        device=device,
    )

    history_path = run_dir / "history.csv"
    history = [
        dict(row)
        for row in _read_csv_rows(history_path)
        if int(row.get("epoch", 0)) <= int(resume_epoch)
    ]
    start = time.perf_counter()
    n = int(x.shape[0])
    batch_size = max(1, int(cfg.train_batch_size))
    for epoch in range(int(resume_epoch), int(cfg.train_epochs)):
        epoch_generator = torch.Generator(device=str(device))
        epoch_generator.manual_seed(_stable_seed(seed, "objective_order", epoch))
        order = torch.randperm(n, generator=epoch_generator, device=device)
        rows: list[dict[str, float]] = []
        net.train()
        for batch_start in range(0, n, batch_size):
            idx = order[batch_start : batch_start + batch_size]
            xb = x.index_select(0, idx)
            yb = y.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True)
            candidates, objective_target, objective_reference, group_size = _objective_training_candidates(
                net,
                xb,
                yb,
                objective=objective_cfg,
                seed=int(seed),
                epoch=int(epoch),
                batch_start=int(batch_start),
                torch=torch,
                device=device,
            )
            predictions = _surrogate_predict_for_objective(surrogate, candidates)
            secondary_predictions = None
            secondary_primary_predictions = None
            if secondary_surrogate is not None:
                cross_per_condition = max(
                    1,
                    min(
                        group_size,
                        int(objective_cfg.get("cross_surrogate_candidates_per_condition", group_size)),
                    ),
                )
                if cross_per_condition == group_size:
                    cross_indices = torch.arange(
                        int(candidates.shape[0]),
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    cross_generator = torch.Generator(device=str(device))
                    cross_generator.manual_seed(
                        _stable_seed(seed, "cross_surrogate", epoch, batch_start)
                    )
                    offsets = torch.stack(
                        [
                            torch.randperm(
                                group_size,
                                generator=cross_generator,
                                device=device,
                            )[:cross_per_condition]
                            for _ in range(int(xb.shape[0]))
                        ],
                        dim=0,
                    )
                    bases = (
                        torch.arange(int(xb.shape[0]), device=device, dtype=torch.long)
                        * group_size
                    ).reshape(-1, 1)
                    cross_indices = (bases + offsets).reshape(-1)
                cross_candidates = candidates.index_select(0, cross_indices)
                secondary_primary_predictions = predictions.index_select(0, cross_indices)
                secondary_predictions = _surrogate_predict_for_objective(
                    secondary_surrogate,
                    cross_candidates,
                )
            batch_objective_cfg = dict(objective_cfg)
            batch_objective_cfg["candidate_group_size"] = int(group_size)
            loss, components = objective_loss_components(
                predictions=predictions,
                secondary_predictions=secondary_predictions,
                secondary_primary_predictions=secondary_primary_predictions,
                target=objective_target,
                candidates=candidates,
                reconstruction=candidates,
                reference=objective_reference,
                torch=torch,
                n_bands=int(bundle.nb),
                objective=batch_objective_cfg,
            )
            loss.backward()
            components["latent_input_gradient_norm"] = float(
                _latent_input_gradient_norm(net, torch=torch).detach().cpu().item()
            )
            mode_count = max(1, int(getattr(net, "mode_count", 1)))
            components["mode_coverage_fraction"] = float(
                min(group_size, mode_count) / mode_count
            )
            optimizer.step()
            rows.append(components)
        epoch_keys = sorted({key for row in rows for key in row})
        epoch_metrics = {key: float(np.mean([row[key] for row in rows])) if rows else 0.0 for key in epoch_keys}
        history.append(
            {
                "stage": "objective_train",
                "seed": int(seed),
                "learning_rate": float(cfg.learning_rate),
                "epoch": int(epoch + 1),
                **epoch_metrics,
            }
        )
        _write_history(history_path, history)
        if int(epoch + 1) in set(cfg.checkpoint_epochs):
            _torch_save_atomic(
                {
                    "model_name": cfg.model_name,
                    "generator_kind": _generator_kind(cfg),
                    "seed": int(seed),
                    "condition_dim": int(x.shape[1]),
                    "candidate_dim": int(y.shape[1]),
                    "latent_dim": int(cfg.latent_dim),
                    "hidden_dim": int(cfg.hidden_dim),
                    "depth": int(cfg.depth),
                    "mode_count": int(cfg.mode_count),
                    "learning_rate": float(cfg.learning_rate),
                    "optimizer": cfg.optimizer,
                    "weight_decay": float(cfg.weight_decay),
                    "state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataset_path": str(cfg.dataset_path),
                    "condition_target": cfg.condition_target,
                    "output_repr": cfg.output_repr,
                    "objective": dict(objective_cfg),
                    "validation_boundary": "inverse_objective_training_surrogate_side_no_mpb_meep",
                    "epoch": int(epoch + 1),
                },
                run_dir / f"objective_checkpoint_epoch_{int(epoch + 1)}_{suffix}.pt",
                torch=torch,
            )
    synchronize_device(device, torch_module=torch)
    train_time_s = float(time.perf_counter() - start)
    out_checkpoint = run_dir / f"objective_model_{suffix}.pt"
    _torch_save_atomic(
        {
            "model_name": cfg.model_name,
            "generator_kind": _generator_kind(cfg),
            "seed": int(seed),
            "condition_dim": int(x.shape[1]),
            "candidate_dim": int(y.shape[1]),
            "latent_dim": int(cfg.latent_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "depth": int(cfg.depth),
            "mode_count": int(cfg.mode_count),
            "learning_rate": float(cfg.learning_rate),
            "optimizer": cfg.optimizer,
            "weight_decay": float(cfg.weight_decay),
            "state_dict": net.state_dict(),
            "dataset_path": str(cfg.dataset_path),
            "condition_target": cfg.condition_target,
            "output_repr": cfg.output_repr,
            "objective": dict(objective_cfg),
            "validation_boundary": "inverse_objective_training_surrogate_side_no_mpb_meep",
        },
        out_checkpoint,
        torch=torch,
    )
    metrics = {
        "seed": int(seed),
        "learning_rate": float(cfg.learning_rate),
        "train_time_s": train_time_s,
        "final_objective_total_loss": float(history[-1]["objective_total_loss"] if history else 0.0),
        "final_surrogate_objective_loss": float(history[-1]["surrogate_objective_loss"] if history else 0.0),
        "final_cross_surrogate_disagreement_bands_2_10": float(
            history[-1].get("cross_surrogate_disagreement_bands_2_10", 0.0) if history else 0.0
        ),
        "final_gap_descriptor_error": float(history[-1].get("gap_descriptor_error", 0.0) if history else 0.0),
        "final_diversity_hamming": float(history[-1].get("diversity_hamming", 0.0) if history else 0.0),
        "final_diversity_loss": float(history[-1].get("diversity_loss", 0.0) if history else 0.0),
        "final_binary_diversity_hamming": float(history[-1].get("binary_diversity_hamming", 0.0) if history else 0.0),
        "final_binary_diversity_loss": float(history[-1].get("binary_diversity_loss", 0.0) if history else 0.0),
        "final_latent_response_std": float(history[-1].get("latent_response_std", 0.0) if history else 0.0),
        "final_latent_response_loss": float(history[-1].get("latent_response_loss", 0.0) if history else 0.0),
        "final_latent_input_gradient_norm": float(
            history[-1].get("latent_input_gradient_norm", 0.0) if history else 0.0
        ),
        "final_mode_coverage_fraction": float(
            history[-1].get("mode_coverage_fraction", 0.0) if history else 0.0
        ),
        "final_quality_weighted_collision_loss": float(
            history[-1].get("quality_weighted_collision_loss", 0.0) if history else 0.0
        ),
        "final_nearest_neighbor_hamming_min": float(history[-1].get("nearest_neighbor_hamming_min", 0.0) if history else 0.0),
        "final_nearest_neighbor_hamming_p10": float(history[-1].get("nearest_neighbor_hamming_p10", 0.0) if history else 0.0),
        "final_nearest_neighbor_hamming_mean": float(
            history[-1].get("nearest_neighbor_hamming_mean", 0.0) if history else 0.0
        ),
        "final_reconstruction_loss": float(history[-1]["reconstruction_loss"] if history else 0.0),
        "checkpoint_path": str(out_checkpoint),
        "resumed_from_checkpoint": resume_checkpoint is not None,
        "resumed_checkpoint_epoch": int(resume_epoch),
    }
    return out_checkpoint, history, metrics


def run_inverse_objective_train(config_path: str | Path) -> Path:
    cfg = load_inverse_run_config(config_path)
    torch = _require_torch()
    device_info = _resolve_runtime_device(cfg, torch)
    run_dir = make_run_dir(cfg.experiment_id, runs_dir=cfg.runs_dir, run_dir=cfg.run_dir)
    run_start = datetime.now().astimezone()
    run_meta = {
        "script": "phc2d inverse objective-train",
        "config_path": str(cfg.source_path),
        "dataset_path": str(cfg.dataset_path),
        "experiment_id": cfg.experiment_id,
        "validation_boundary": "inverse_objective_training_surrogate_side_no_mpb_meep",
        "pid": os.getpid(),
        "start_timestamp": run_start.isoformat(),
        **collect_device_metadata(device_info, cfg.runtime),
    }
    logger = setup_logging(log_file=run_dir / "run.log", run_meta=run_meta)
    write_resolved_config(run_dir, cfg.to_resolved_yaml())
    ensure_figures_dir(run_dir)
    reset_peak_gpu_memory(device_info.torch_device, torch_module=torch)

    wall_start = time.perf_counter()
    history: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    learning_rates = _configured_learning_rates(cfg)
    for learning_rate in learning_rates:
        for seed in cfg.seeds:
            checkpoint, rows, metrics = _train_one_objective_generator(
                replace(cfg, seed=int(seed), learning_rate=float(learning_rate), seeds=(int(seed),)),
                seed=int(seed),
                run_dir=run_dir,
                torch=torch,
                device_info=device_info,
            )
            history.extend(rows)
            seed_rows.append(metrics)
            checkpoints.append(str(checkpoint))
    if checkpoints:
        latest_payload = torch.load(checkpoints[0], map_location=device_info.torch_device, weights_only=False)
        torch.save(latest_payload, run_dir / "model.pt")

    train_time_s = float(sum(float(row["train_time_s"]) for row in seed_rows))
    metrics_payload = {
        "status": "ok",
        "stage": "inverse_objective_train",
        "validation_boundary": "inverse_objective_training_surrogate_side_no_mpb_meep",
        "surrogate_side_no_mpb_meep": True,
        "seeds": [int(seed) for seed in cfg.seeds],
        "n_seeds": int(len(cfg.seeds)),
        "learning_rates": [float(item) for item in learning_rates],
        "n_learning_rates": int(len(learning_rates)),
        "train_time_s": train_time_s,
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "final_objective_total_loss_mean": float(np.mean([float(row["final_objective_total_loss"]) for row in seed_rows]))
        if seed_rows
        else 0.0,
        "final_surrogate_objective_loss_mean": float(
            np.mean([float(row["final_surrogate_objective_loss"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_cross_surrogate_disagreement_bands_2_10_mean": float(
            np.mean([float(row["final_cross_surrogate_disagreement_bands_2_10"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_gap_descriptor_error_mean": float(
            np.mean([float(row["final_gap_descriptor_error"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_diversity_hamming_mean": float(np.mean([float(row["final_diversity_hamming"]) for row in seed_rows]))
        if seed_rows
        else 0.0,
        "final_diversity_loss_mean": float(np.mean([float(row["final_diversity_loss"]) for row in seed_rows]))
        if seed_rows
        else 0.0,
        "final_binary_diversity_hamming_mean": float(
            np.mean([float(row["final_binary_diversity_hamming"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_binary_diversity_loss_mean": float(
            np.mean([float(row["final_binary_diversity_loss"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_latent_response_std_mean": float(np.mean([float(row["final_latent_response_std"]) for row in seed_rows]))
        if seed_rows
        else 0.0,
        "final_latent_response_loss_mean": float(np.mean([float(row["final_latent_response_loss"]) for row in seed_rows]))
        if seed_rows
        else 0.0,
        "final_latent_input_gradient_norm_mean": float(
            np.mean([float(row["final_latent_input_gradient_norm"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_mode_coverage_fraction_mean": float(
            np.mean([float(row["final_mode_coverage_fraction"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_quality_weighted_collision_loss_mean": float(
            np.mean([float(row["final_quality_weighted_collision_loss"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_nearest_neighbor_hamming_min_mean": float(
            np.mean([float(row["final_nearest_neighbor_hamming_min"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_nearest_neighbor_hamming_p10_mean": float(
            np.mean([float(row["final_nearest_neighbor_hamming_p10"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "final_nearest_neighbor_hamming_mean_mean": float(
            np.mean([float(row["final_nearest_neighbor_hamming_mean"]) for row in seed_rows])
        )
        if seed_rows
        else 0.0,
        "checkpoint_paths": checkpoints,
        **_device_metadata_with_peak(device_info, torch),
    }
    _write_history(run_dir / "history.csv", history)
    _write_history(run_dir / "seed_metrics.csv", seed_rows)
    write_metrics(run_dir, metrics_payload)
    write_results(
        run_dir,
        {
            "action": "inverse objective-train",
            "stage": "inverse_objective_train",
            "run_dir": str(run_dir),
            "experiment_id": cfg.experiment_id,
            "dataset_path": str(cfg.dataset_path),
            "checkpoint_paths": checkpoints,
            "learning_rates": [float(item) for item in learning_rates],
            "surrogate_side_no_mpb_meep": True,
            "mpb_meep_validation": False,
        },
    )
    logger.info("Inverse objective train finished: %s", run_dir)
    run_meta.update(metrics_payload)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    return run_dir


def _replay_one_target(
    cfg: GenerativeInverseConfig,
    target_payload: Mapping[str, Any],
    target: np.ndarray,
    target_id: str,
    target_index: int,
    *,
    torch: Any,
    device_info: ResolvedDeviceInfo,
) -> dict[str, Any]:
    candidates = np.asarray(target_payload.get("candidates", []), dtype=np.float32)
    if candidates.size == 0:
        candidates = np.asarray(target_payload.get("raw_candidates", []), dtype=np.float32)
    if candidates.size == 0:
        raise ValueError(f"Target {target_id} has no candidates")

    selected_indices = [int(item) for item in target_payload.get("selected_indices", [])][: cfg.surrogate_top_m]
    if not selected_indices:
        selected_indices = list(range(min(cfg.surrogate_top_m, int(candidates.shape[0]))))
    selected = candidates[selected_indices]
    if selected.ndim != 2:
        selected = selected.reshape(selected.shape[0], -1)

    surrogate = _MockSurrogate(
        input_dim=int(selected.shape[1]),
        target_shape=tuple(target.shape),
        seed=_surrogate_seed(cfg, target_id, target_index),
        torch=torch,
        device=device_info.torch_device,
    )
    selected_tensor = torch.as_tensor(selected, dtype=torch.float32, device=device_info.torch_device)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device_info.torch_device).reshape(1, -1)
    synchronize_device(device_info.torch_device, torch_module=torch)
    surrogate_pred_tensor = surrogate.predict_tensor(selected_tensor)
    synchronize_device(device_info.torch_device, torch_module=torch)
    surrogate_pred = surrogate_pred_tensor.detach().cpu().numpy().astype(np.float32)
    target_flat = target_tensor.expand_as(surrogate_pred_tensor.reshape(int(surrogate_pred_tensor.shape[0]), -1))
    surrogate_errors_tensor = torch.abs(surrogate_pred_tensor.reshape(int(surrogate_pred_tensor.shape[0]), -1) - target_flat) / torch.clamp(torch.abs(target_flat), min=1.0e-8)
    surrogate_errors = surrogate_errors_tensor.mean(dim=1).detach().cpu().numpy().astype(np.float32)

    replay_rng = np.random.default_rng(_replay_seed(cfg, target_id, target_index))
    noise_scale = float(cfg.ranking.get("replay_noise", cfg.threshold / 2.0))
    replay_pred = surrogate_pred + noise_scale * replay_rng.normal(size=surrogate_pred.shape)
    replay_errors = _relative_mre(replay_pred, target).reshape(replay_pred.shape[0], -1).mean(axis=1)

    return {
        "selected_indices": selected_indices,
        "best_of_k_surrogate_mre": float(np.min(surrogate_errors)),
        "best_of_k_replay_mre": float(np.min(replay_errors)),
        "surrogate_replay_gap": float(np.min(replay_errors) - np.min(surrogate_errors)),
        "surrogate_errors": _to_jsonable(surrogate_errors),
        "replay_errors": _to_jsonable(replay_errors),
        "mpb_replay_time_per_candidate_s": 0.0,
    }


def run_inverse_sample(config_path: str | Path) -> Path:
    cfg = _prepare_sample_config(load_inverse_run_config(config_path))
    torch = _require_torch()
    device_info = _resolve_runtime_device(cfg, torch)
    run_dir = make_run_dir(cfg.experiment_id, runs_dir=cfg.runs_dir, run_dir=cfg.run_dir)
    run_start = datetime.now().astimezone()
    run_meta = {
        "script": "phc2d inverse sample",
        "config_path": str(cfg.source_path),
        "dataset_path": str(cfg.dataset_path),
        "experiment_id": cfg.experiment_id,
        "validation_boundary": "inverse_surrogate_side_no_mpb_meep"
        if not cfg.allow_mock_surrogate
        else "inverse_chain_smoke",
        "pid": os.getpid(),
        "start_timestamp": run_start.isoformat(),
        **collect_device_metadata(device_info, cfg.runtime),
    }
    logger = setup_logging(
        log_file=run_dir / "run.log",
        run_meta=run_meta,
    )
    write_resolved_config(run_dir, cfg.to_resolved_yaml())
    ensure_figures_dir(run_dir)

    wall_start = time.perf_counter()
    target_records, metrics = _run_sample(cfg, run_dir, torch=torch, device_info=device_info)
    _write_history(
        run_dir / "history.csv",
        [
            {
                "stage": "sample",
                "target_id": item["target_id"],
                "target_index": int(item["target_index"]),
                "candidate_count_raw": float(
                    len(item.get("raw_candidates", [])) if isinstance(item.get("raw_candidates"), list) else 0
                ),
                "candidate_count_filtered": float(
                    len(item.get("candidates", [])) if isinstance(item.get("candidates"), list) else 0
                ),
                "best_of_k_surrogate_mre": float(item.get("surrogate_metrics", {}).get("best_of_k_surrogate_mre", 0.0)),
            }
            for item in target_records
        ],
    )
    write_metrics(run_dir, metrics)
    write_results(
        run_dir,
        {
            "action": "inverse sample",
            "stage": cfg.stage,
            "run_dir": str(run_dir),
            "experiment_id": cfg.experiment_id,
            "dataset_path": str(cfg.dataset_path),
        },
    )
    try:
        plot_metric_report(metrics, run_dir / "figures" / "metrics.png", title="inverse sample metrics")
    except Exception:
        logger.warning("Could not write inverse sample metric figure.")
    logger.info("Inverse sample finished: %s", run_dir)
    run_meta.update(metrics)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    return run_dir


def run_inverse_eval(config_path: str | Path, *, selected_only: bool = True) -> Path:
    cfg = load_inverse_run_config(config_path)
    if not cfg.replay_enabled:
        raise ValueError(
            "`phc2d inverse eval` writes replay metrics and requires replay.enabled=true. "
            "Use `phc2d inverse sample` for surrogate-only inverse results."
        )
    torch = _require_torch()
    device_info = _resolve_runtime_device(cfg, torch)
    run_dir = _resolve_sample_run(cfg)
    if run_dir is None:
        run_dir = run_inverse_sample(config_path)

    run_start = datetime.now().astimezone()
    run_meta = {
        "script": "phc2d inverse eval",
        "config_path": str(cfg.source_path),
        "dataset_path": str(cfg.dataset_path),
        "experiment_id": cfg.experiment_id,
        "validation_boundary": "inverse_chain_smoke",
        "pid": os.getpid(),
        "start_timestamp": run_start.isoformat(),
        **collect_device_metadata(device_info, cfg.runtime),
    }
    logger = setup_logging(
        log_file=run_dir / "run.log",
        run_meta=run_meta,
    )
    ensure_figures_dir(run_dir)

    payload_path = run_dir / "candidates.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    targets = [dict(item) for item in payload.get("targets", [])]
    if not targets:
        raise ValueError(f"No candidates found in {payload_path}")

    wall_start = time.perf_counter()
    replay_rows: list[dict[str, Any]] = []
    for item in targets:
        target_id = str(item.get("target_id", "unknown"))
        target_index = int(item.get("target_index", 0))
        condition = dict(item.get("condition", {}))
        target = np.asarray(condition.get(cfg.condition_target, []), dtype=np.float32)
        if target.size == 0:
            continue

        eval_start = time.perf_counter()
        replay_payload = _replay_one_target(
            cfg,
            target_payload=item,
            target=target,
            target_id=target_id,
            target_index=target_index,
            torch=torch,
            device_info=device_info,
        )
        wall_time = float(time.perf_counter() - eval_start)
        replay_payload["mpb_replay_time_s"] = wall_time
        replay_payload["mpb_replay_time_per_candidate_s"] = wall_time / max(
            len(replay_payload.get("selected_indices", [])), 1
        )

        if selected_only:
            item["selected_indices"] = replay_payload.get("selected_indices", [])
        item["replay"] = replay_payload
        replay_rows.append(
            {
                "target_id": target_id,
                "target_index": target_index,
                "best_of_k_surrogate_mre": float(replay_payload["best_of_k_surrogate_mre"]),
                "best_of_k_replay_mre": float(replay_payload["best_of_k_replay_mre"]),
                "surrogate_replay_gap": float(replay_payload["surrogate_replay_gap"]),
                "mpb_replay_time_s": float(wall_time),
            }
        )

    if replay_rows:
        best_surrogate = float(np.mean([row["best_of_k_surrogate_mre"] for row in replay_rows]))
        best_replay = float(np.mean([row["best_of_k_replay_mre"] for row in replay_rows]))
        replay_gap = float(np.mean([row["surrogate_replay_gap"] for row in replay_rows]))
        mpb_time = float(sum(row["mpb_replay_time_s"] for row in replay_rows))
    else:
        best_surrogate = 0.0
        best_replay = 0.0
        replay_gap = 0.0
        mpb_time = 0.0

    metrics = {
        "best_of_k_surrogate_mre": best_surrogate,
        "best_of_k_replay_mre": best_replay,
        "surrogate_replay_gap": replay_gap,
        "mpb_replay_time_s": mpb_time,
        "mpb_replay_time_per_candidate_s": float(mpb_time / max(len(replay_rows), 1)),
        "wall_time_total_s": float(time.perf_counter() - wall_start),
        "batch_size": int(cfg.surrogate_top_m),
        "effective_batch_size": int(cfg.surrogate_top_m),
        **_device_metadata_with_peak(device_info, torch),
    }
    write_metrics(run_dir, metrics)
    write_results(
        run_dir,
        {
            "action": "inverse eval",
            "stage": "replay",
            "run_dir": str(run_dir),
            "targets": replay_rows,
            "replay_enabled": bool(cfg.replay_enabled),
        },
    )
    _write_history(
        run_dir / "history.csv",
        [
            {
                "stage": "eval",
                "target_id": row["target_id"],
                "target_index": int(row["target_index"]),
                "mpb_replay_time_s": float(row["mpb_replay_time_s"]),
                "surrogate_replay_gap": float(row["surrogate_replay_gap"]),
            }
            for row in replay_rows
        ],
    )
    payload_path.write_text(
        json.dumps({"version": 1, "targets": targets}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        plot_metric_report(metrics, run_dir / "figures" / "metrics.png", title="inverse eval metrics")
    except Exception:
        logger.warning("Could not write inverse eval metric figure.")
    logger.info("Inverse eval finished: %s", run_dir)
    run_meta.update(metrics)
    _finalize_run_meta(run_dir, run_meta, wall_time_s=float(time.perf_counter() - wall_start))
    return run_dir


__all__ = [
    "run_inverse_train",
    "run_inverse_objective_train",
    "run_inverse_sample",
    "run_inverse_rerank",
    "run_inverse_eval",
]
