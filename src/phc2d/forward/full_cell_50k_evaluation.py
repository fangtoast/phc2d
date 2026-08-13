"""Run the frozen post-training evaluation campaign for the 50k full-cell study.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell import _FullCellH5Dataset
from phc2d.forward.full_cell_cross_eval import THRESHOLDS, WAVE_LABELS
from phc2d.forward.full_cell_wave import MODELS, SEEDS
from phc2d.models import build_model
from phc2d.training import DeviceConfig, resolve_device

COMPATIBLE_RESOLUTION_MODELS = {"lifted_cnn", "fno2d_band_v2"}
EXPECTED_LEGACY_MAIN = 84
EXPECTED_LEGACY_ABLATION = 18
EXPECTED_CURRENT_40K = 72
EXPECTED_EXECUTABLE_CELLS = 906
EXPECTED_NOT_APPLICABLE_CELLS = 156
EXPECTED_TOTAL_CELLS = EXPECTED_EXECUTABLE_CELLS + EXPECTED_NOT_APPLICABLE_CELLS


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(fields or [])
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    if not keys:
        raise ValueError(f"CSV schema is required for empty rows: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON mapping: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_values(values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _representation_label(value: str) -> str:
    aliases = {"binary_native": "binary", "graded_continuous": "continuous"}
    label = aliases.get(str(value), str(value))
    if label not in WAVE_LABELS:
        raise ValueError(f"unsupported representation: {value}")
    return label


def _is_validation_checkpoint_result(results: Mapping[str, Any]) -> bool:
    """Accept pre-role-field runs only when the immutable split facts agree."""

    return (
        results.get("selection_split") == "validation"
        and not bool(results.get("test_executed"))
        and results.get("checkpoint_role", "best_validation") == "best_validation"
    )


def _checkpoint_record(root: Path, run_dir: Path) -> tuple[Path, str, dict[str, Any]]:
    results = _read_json(run_dir / "results.json")
    if results.get("status") != "formal_training_success":
        raise RuntimeError(f"checkpoint source is not a successful formal run: {run_dir}")
    checkpoint = _resolve(root, str(results.get("checkpoint_path") or run_dir / "model.pt"))
    if not checkpoint.is_file():
        checkpoint = run_dir / "model.pt"
    expected_hash = str(results.get("checkpoint_sha256", ""))
    if len(expected_hash) != 64:
        raise RuntimeError(f"checkpoint source lacks a frozen SHA-256: {run_dir}")
    # The plan consumes the hash already frozen by the successful training run.
    # Actual checkpoint bytes are re-hashed once, in parallel, by the shard that
    # owns the complete source bundle.
    return checkpoint, expected_hash, results


def _source_id(row: Mapping[str, Any]) -> str:
    parts = (
        row["source_group"],
        row["source_representation"],
        row["model"],
        row["seed"],
        row["epoch_budget"],
        row["training_resolution"],
        row["checkpoint_sha256"],
    )
    return "src_" + hashlib.sha256("|".join(str(value) for value in parts).encode("utf-8")).hexdigest()[:20]


def _load_sources(
    *,
    root: Path,
    legacy_main_manifest: Path,
    legacy_ablation_manifest: Path,
    training_campaign_status: Path,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for row in _read_csv(legacy_main_manifest):
        run_dir = _resolve(root, row["run_dir"])
        checkpoint, checkpoint_hash, results = _checkpoint_record(root, run_dir)
        source = {
            "source_group": "legacy10k_main",
            "source_dataset_size": 10000,
            "source_representation": _representation_label(row["wave"]),
            "model": row["model"],
            "seed": int(row["seed"]),
            "epoch_budget": 100,
            "training_resolution": 64,
            "checkpoint_role": "best_validation",
            "checkpoint_path": _relative(root, checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "run_dir": _relative(root, run_dir),
            "selection_split": results.get("selection_split", "validation"),
        }
        source["source_id"] = _source_id(source)
        sources.append(source)
    if len(sources) != EXPECTED_LEGACY_MAIN:
        raise RuntimeError(f"legacy main inventory must contain {EXPECTED_LEGACY_MAIN} sources")

    ablation_sources: list[dict[str, Any]] = []
    for row in _read_csv(legacy_ablation_manifest):
        run_dir = _resolve(root, row["run_dir"])
        checkpoint, checkpoint_hash, results = _checkpoint_record(root, run_dir)
        source = {
            "source_group": "legacy10k_ablation",
            "source_dataset_size": 10000,
            "source_representation": _representation_label(row["representation"]),
            "model": row["model"],
            "seed": int(row["seed"]),
            "epoch_budget": 100,
            "training_resolution": int(row["resolution"]),
            "checkpoint_role": "best_validation_native_resolution",
            "checkpoint_path": _relative(root, checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "run_dir": _relative(root, run_dir),
            "selection_split": results.get("selection_split", "validation"),
        }
        source["source_id"] = _source_id(source)
        ablation_sources.append(source)
    if len(ablation_sources) != EXPECTED_LEGACY_ABLATION:
        raise RuntimeError(f"legacy ablation inventory must contain {EXPECTED_LEGACY_ABLATION} sources")
    sources.extend(ablation_sources)

    campaign = _read_json(training_campaign_status)
    current_sources: list[dict[str, Any]] = []
    for wave in campaign.get("waves", []):
        if int(wave.get("train_size", 0)) != 40000:
            continue
        representation = _representation_label(str(wave["representation"]))
        if representation not in {"binary", "continuous"}:
            continue
        manifest = _resolve(root, wave["manifest"])
        for row in _read_csv(manifest):
            run_dir = _resolve(root, row["run_dir"])
            checkpoint, checkpoint_hash, results = _checkpoint_record(root, run_dir)
            if not _is_validation_checkpoint_result(results):
                raise RuntimeError(f"40k evaluation source is not validation-selected: {run_dir}")
            source = {
                "source_group": "current40k",
                "source_dataset_size": 40000,
                "source_representation": representation,
                "model": row["model"],
                "seed": int(row["seed"]),
                "epoch_budget": int(wave["max_epochs"]),
                "training_resolution": int(wave["resolution"]),
                "checkpoint_role": "best_validation",
                "checkpoint_path": _relative(root, checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "run_dir": _relative(root, run_dir),
                "selection_split": results.get("selection_split", "validation"),
            }
            source["source_id"] = _source_id(source)
            current_sources.append(source)
    if len(current_sources) != EXPECTED_CURRENT_40K:
        raise RuntimeError(f"40k inventory must contain {EXPECTED_CURRENT_40K} sources")
    sources.extend(current_sources)
    ids = [str(row["source_id"]) for row in sources]
    if len(ids) != len(set(ids)):
        raise RuntimeError("source inventory contains duplicate identities")
    return sources


def build_evaluation_cells(sources: Sequence[Mapping[str, Any]], *, shard_count: int) -> list[dict[str, Any]]:
    """Build the frozen, de-duplicated 50k evaluation matrix."""

    if int(shard_count) < 1:
        raise ValueError("shard_count must be positive")
    by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}

    def add(source: Mapping[str, Any], target: str, resolution: int, contract: str, *, status: str = "planned") -> None:
        target = _representation_label(target)
        key = (str(source["source_id"]), target, int(resolution), status)
        if key not in by_key:
            base = dict(source)
            base.update(
                {
                    "target_representation": target,
                    "target_lane": "binary" if target == "binary" else "graded",
                    "evaluation_resolution": int(resolution),
                    "planned_status": status,
                    "contracts": set(),
                }
            )
            by_key[key] = base
        by_key[key]["contracts"].add(contract)

    for source in sources:
        group = str(source["source_group"])
        source_representation = str(source["source_representation"])
        training_resolution = int(source["training_resolution"])
        if group == "legacy10k_main":
            for target in WAVE_LABELS:
                add(source, target, 64, "legacy10k_to_50k_native" if target == source_representation else "legacy10k_to_50k_cross")
        elif group == "legacy10k_ablation":
            add(source, source_representation, training_resolution, "legacy10k_to_50k_native")
        elif group == "current40k":
            for target in ("binary", "continuous"):
                add(source, target, 64, "current40k_native" if target == source_representation else "current40k_cross")

        if group in {"legacy10k_main", "current40k"}:
            if str(source["model"]) in COMPATIBLE_RESOLUTION_MODELS:
                for resolution in (32, 64, 128):
                    add(source, source_representation, resolution, "zero_shot_input_resolution_transfer")
            else:
                for resolution in (32, 128):
                    add(
                        source,
                        source_representation,
                        resolution,
                        "zero_shot_input_resolution_transfer",
                        status="not_applicable_architecture_shape_mismatch",
                    )

    executable_load: dict[str, int] = defaultdict(int)
    for row in by_key.values():
        if row["planned_status"] == "planned":
            executable_load[str(row["source_id"])] += 1
    shard_loads = [0] * int(shard_count)
    shard_for_source: dict[str, int] = {}
    for source_id, weight in sorted(executable_load.items(), key=lambda item: (-item[1], item[0])):
        shard = min(range(int(shard_count)), key=lambda index: (shard_loads[index], index))
        shard_for_source[source_id] = shard
        shard_loads[shard] += weight

    rows: list[dict[str, Any]] = []
    ordered = sorted(
        by_key.values(),
        key=lambda row: (
            str(row["source_group"]),
            str(row["source_representation"]),
            int(row["epoch_budget"]),
            str(row["model"]),
            int(row["seed"]),
            str(row["target_representation"]),
            int(row["evaluation_resolution"]),
            str(row["planned_status"]),
        ),
    )
    for plan_index, row in enumerate(ordered):
        contracts = "|".join(sorted(row.pop("contracts")))
        evaluation_id = "eval_" + hashlib.sha256(
            f"{row['source_id']}|{row['target_representation']}|{row['evaluation_resolution']}|{row['planned_status']}".encode("utf-8")
        ).hexdigest()[:20]
        row.update(
            {
                "plan_index": plan_index,
                "evaluation_id": evaluation_id,
                "contracts": contracts,
                "shard_id": shard_for_source.get(str(row["source_id"]), ""),
            }
        )
        rows.append(row)
    executable = sum(row["planned_status"] == "planned" for row in rows)
    not_applicable = sum(row["planned_status"] != "planned" for row in rows)
    if executable != EXPECTED_EXECUTABLE_CELLS or not_applicable != EXPECTED_NOT_APPLICABLE_CELLS:
        raise RuntimeError(
            f"evaluation matrix mismatch: executable={executable}, not_applicable={not_applicable}"
        )
    return rows


def plan_evaluation_campaign(
    *,
    campaign_dir: str | Path,
    legacy_main_manifest: str | Path,
    legacy_ablation_manifest: str | Path,
    training_campaign_status: str | Path,
    binary_dataset: str | Path,
    graded_dataset: str | Path,
    repository_root: str | Path = ".",
    shard_count: int = 8,
    bootstrap_replicates: int = 500,
) -> Path:
    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError(f"campaign directory is not empty: {campaign}")
    campaign.mkdir(parents=True, exist_ok=True)
    paths = {
        "legacy_main_manifest": _resolve(root, legacy_main_manifest),
        "legacy_ablation_manifest": _resolve(root, legacy_ablation_manifest),
        "training_campaign_status": _resolve(root, training_campaign_status),
        "binary_dataset": _resolve(root, binary_dataset),
        "graded_dataset": _resolve(root, graded_dataset),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    sources = _load_sources(
        root=root,
        legacy_main_manifest=paths["legacy_main_manifest"],
        legacy_ablation_manifest=paths["legacy_ablation_manifest"],
        training_campaign_status=paths["training_campaign_status"],
    )
    rows = build_evaluation_cells(sources, shard_count=int(shard_count))
    _write_csv(campaign / "checkpoint_inventory.csv", sources)
    _write_csv(
        campaign / "legacy_checkpoint_manifest.csv",
        [row for row in sources if str(row["source_group"]).startswith("legacy10k")],
    )
    training_campaign = _read_json(paths["training_campaign_status"])
    _write_csv(
        campaign / "training_wave_manifest.csv",
        [
            {
                "index": wave["index"],
                "key": wave["key"],
                "representation": wave["representation"],
                "train_size": wave["train_size"],
                "max_epochs": wave["max_epochs"],
                "resolution": wave["resolution"],
                "status": wave["status"],
                "planned_runs": wave["planned_runs"],
                "manifest": wave["manifest"],
                "aggregation_dir": wave["aggregation_dir"],
                "job_ids": "|".join(str(value) for value in wave["job_ids"]),
            }
            for wave in training_campaign["waves"]
        ],
    )
    _write_csv(campaign / "evaluation_plan.csv", rows)
    _write_csv(
        campaign / "cross_evaluation_manifest.csv",
        [row for row in rows if "cross" in str(row["contracts"])],
    )
    dataset_hashes = {
        "binary": _sha256_file(paths["binary_dataset"]),
        "graded": _sha256_file(paths["graded_dataset"]),
    }
    dataset_sizes = {
        "binary": paths["binary_dataset"].stat().st_size,
        "graded": paths["graded_dataset"].stat().st_size,
    }
    plan_hash = _sha256_values(
        [
            {
                key: row[key]
                for key in (
                    "evaluation_id",
                    "source_id",
                    "target_representation",
                    "evaluation_resolution",
                    "planned_status",
                    "contracts",
                    "shard_id",
                )
            }
            for row in rows
        ]
    )
    resolved = {
        "schema_version": "phc2d.full_cell_50k_evaluation.v1",
        "paths": {key: _relative(root, value) for key, value in paths.items()},
        "dataset_sha256": dataset_hashes,
        "dataset_size_bytes": dataset_sizes,
        "shard_count": int(shard_count),
        "bootstrap_replicates": int(bootstrap_replicates),
        "plan_hash": plan_hash,
        "expected": {
            "checkpoint_sources": len(sources),
            "legacy_checkpoint_sources": EXPECTED_LEGACY_MAIN + EXPECTED_LEGACY_ABLATION,
            "current_40k_sources": EXPECTED_CURRENT_40K,
            "executable_cells": EXPECTED_EXECUTABLE_CELLS,
            "not_applicable_cells": EXPECTED_NOT_APPLICABLE_CELLS,
            "total_cells": EXPECTED_TOTAL_CELLS,
        },
        "scientific_boundary": {
            "legacy_10k_checkpoints": "frozen_no_finetune_source_normalization",
            "current_40k_checkpoints": "best_validation_disjoint_50k_test",
            "current_50k_final_fit": "excluded_no_unbiased_test_available",
            "resolution_transfer": "zero_shot_input_resolution_not_retraining_or_physical_convergence",
        },
    }
    (campaign / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        campaign / "campaign_status.json",
        {
            "status": "evaluation_plan_ready",
            "plan_hash": plan_hash,
            "shard_count": int(shard_count),
            "planned_cells": EXPECTED_EXECUTABLE_CELLS,
            "not_applicable_cells": EXPECTED_NOT_APPLICABLE_CELLS,
            "completed_cells": 0,
            "failed_cells": 0,
            "shard_job_id": None,
            "finalizer_job_id": None,
            "next_gate": "authorized_slurm_evaluation_submission",
        },
    )
    return campaign


def _evaluate_samples(model: Any, loader: Any, *, device: Any, mean: Any, std: Any, torch: Any) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    sample_mre: list[np.ndarray] = []
    sample_mae: list[np.ndarray] = []
    sample_mse: list[np.ndarray] = []
    normalized_squared = 0.0
    count = 0
    started = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            prediction_norm = model(xb)
            normalized_squared += float(torch.sum((prediction_norm - yb) ** 2).detach().cpu())
            prediction = prediction_norm * std + mean
            target = yb * std + mean
            difference = prediction - target
            sample_mre.append(
                (100.0 * torch.mean(torch.abs(difference[..., 1:]) / torch.clamp(torch.abs(target[..., 1:]), min=1.0e-8), dim=(1, 2))).detach().cpu().numpy()
            )
            sample_mae.append(torch.mean(torch.abs(difference), dim=(1, 2)).detach().cpu().numpy())
            sample_mse.append(torch.mean(difference * difference, dim=(1, 2)).detach().cpu().numpy())
            count += int(target.numel())
    mre = np.concatenate(sample_mre).astype(np.float32, copy=False)
    mae = np.concatenate(sample_mae).astype(np.float32, copy=False)
    mse = np.concatenate(sample_mse).astype(np.float32, copy=False)
    metrics = {
        "bands_2_10_mre_pct": float(np.mean(mre)),
        "mae": float(np.mean(mae)),
        "rmse": float(np.sqrt(np.mean(mse))),
        "normalized_mse": normalized_squared / max(count, 1),
        "inference_time_s": time.perf_counter() - started,
        "sample_count": int(mre.size),
    }
    return metrics, mre, mae, mse


def run_evaluation_shard(
    *,
    campaign_dir: str | Path,
    shard_id: int,
    repository_root: str | Path = ".",
    device: str = "cuda:0",
) -> Path:
    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    resolved = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    plan = _read_csv(campaign / "evaluation_plan.csv")
    selected = [
        row
        for row in plan
        if row["planned_status"] == "planned" and int(row["shard_id"]) == int(shard_id)
    ]
    if not selected:
        raise RuntimeError(f"shard {shard_id} has no executable cells")
    out = campaign / "shards" / f"shard_{int(shard_id):02d}"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / "shard_status.json",
        {
            "status": "running",
            "shard_id": int(shard_id),
            "planned_cells": len(selected),
            "completed_cells": 0,
            "failed_cells": 0,
            "heartbeat_at": time.time(),
        },
    )
    import h5py
    import torch
    from torch.utils.data import DataLoader

    device_info = resolve_device(
        DeviceConfig(
            device=device,
            require_accelerator=device.startswith("cuda"),
            precision="fp32",
            amp_enabled=False,
            tf32_enabled=False,
            num_workers=4,
            pin_memory=True,
        ),
        torch_module=torch,
    )
    dataset_paths = {
        "binary": _resolve(root, resolved["paths"]["binary_dataset"]),
        "graded": _resolve(root, resolved["paths"]["graded_dataset"]),
    }
    dataset_indices: dict[str, np.ndarray] = {}
    dataset_index_hashes: dict[str, str] = {}
    for lane, path in dataset_paths.items():
        if path.stat().st_size != int(resolved["dataset_size_bytes"][lane]):
            raise RuntimeError(f"dataset size changed after the plan hash gate: {path}")
        with h5py.File(path, "r") as handle:
            indices = np.asarray(handle["idx_test"], dtype=np.int64)
        dataset_indices[lane] = indices
        dataset_index_hashes[lane] = hashlib.sha256(indices.tobytes()).hexdigest()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["source_id"]].append(row)
    result_rows: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    sample_mre_rows: list[np.ndarray] = []
    sample_mae_rows: list[np.ndarray] = []
    sample_mse_rows: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        for source_id in sorted(grouped):
            source_rows = sorted(grouped[source_id], key=lambda row: int(row["plan_index"]))
            first = source_rows[0]
            checkpoint_path = _resolve(root, first["checkpoint_path"])
            if _sha256_file(checkpoint_path) != first["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint SHA-256 changed: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device_info.torch_device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=device_info.torch_device)
            if str(checkpoint.get("model_name")) != first["model"]:
                raise RuntimeError(f"checkpoint model identity mismatch: {checkpoint_path}")
            model = build_model(
                first["model"],
                input_shape=tuple(int(value) for value in checkpoint["input_shape"]),
                output_shape=tuple(int(value) for value in checkpoint["output_shape"]),
                **dict(checkpoint["model_params"]),
            ).to(device_info.torch_device)
            model.load_state_dict(checkpoint["state_dict"])
            mean = checkpoint["target_mean"].to(device_info.torch_device)
            std = checkpoint["target_std"].to(device_info.torch_device)
            for row in source_rows:
                lane = row["target_lane"]
                target = row["target_representation"]
                resolution = int(row["evaluation_resolution"])
                layout = "flat" if row["model"] in {"mlp", "res_mlp_wide"} else "image"
                dataset = _FullCellH5Dataset(
                    dataset_paths[lane],
                    dataset_indices[lane],
                    grid_field=f"material_weight_grid_{resolution}",
                    layout=layout,
                    sampling="deterministic_threshold" if target.startswith("t0") else "continuous",
                    threshold=THRESHOLDS.get(target),
                    target_mean=mean.detach().cpu().numpy(),
                    target_std=std.detach().cpu().numpy(),
                )
                loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
                metrics, sample_mre, sample_mae, sample_mse = _evaluate_samples(
                    model,
                    loader,
                    device=device_info.torch_device,
                    mean=mean,
                    std=std,
                    torch=torch,
                )
                if not all(math.isfinite(float(metrics[key])) for key in ("bands_2_10_mre_pct", "mae", "rmse")):
                    raise RuntimeError(f"non-finite evaluation metrics: {row['evaluation_id']}")
                result_rows.append(
                    {
                        **row,
                        "status": "completed",
                        **metrics,
                        "test_indices_sha256": dataset_index_hashes[lane],
                        "source_normalization_reused": True,
                        "fine_tuned": False,
                        "selection_influence": False,
                    }
                )
                sample_ids.append(row["evaluation_id"])
                sample_mre_rows.append(sample_mre)
                sample_mae_rows.append(sample_mae)
                sample_mse_rows.append(sample_mse)
                _write_csv(out / "history.csv", result_rows)
                _write_json(
                    out / "shard_status.json",
                    {
                        "status": "running",
                        "shard_id": int(shard_id),
                        "planned_cells": len(selected),
                        "completed_cells": len(result_rows),
                        "failed_cells": 0,
                        "heartbeat_at": time.time(),
                    },
                )
        _write_csv(out / "evaluation_cells.csv", result_rows)
        npz_tmp = out / "sample_metrics.npz.tmp"
        with npz_tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                evaluation_ids=np.asarray(sample_ids),
                sample_mre_pct=np.stack(sample_mre_rows),
                sample_mae=np.stack(sample_mae_rows),
                sample_mse=np.stack(sample_mse_rows),
            )
        os.replace(npz_tmp, out / "sample_metrics.npz")
        wall_time = time.perf_counter() - started
        _write_json(
            out / "metrics.json",
            {
                "planned_cells": len(selected),
                "completed_cells": len(result_rows),
                "failed_cells": 0,
                "wall_time_total_s": wall_time,
                "inference_time_total_s": sum(float(row["inference_time_s"]) for row in result_rows),
                "throughput_cells_per_s": len(result_rows) / max(wall_time, 1.0e-12),
            },
        )
        _write_json(
            out / "results.json",
            {
                "status": "formal_evaluation_shard_success",
                "shard_id": int(shard_id),
                "completed_cells": len(result_rows),
                "failed_cells": 0,
                "plan_hash": resolved["plan_hash"],
            },
        )
        _write_json(
            out / "run.meta.json",
            {
                "script": "python -m phc2d.forward.full_cell_50k_evaluation run-shard",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
                "deployed_git_commit": os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT"),
                **device_info.metadata(),
            },
        )
        (out / "config.resolved.yaml").write_text(
            yaml.safe_dump({"campaign": _relative(root, campaign), "shard_id": int(shard_id), "device": device}, sort_keys=False),
            encoding="utf-8",
        )
        _write_json(
            out / "shard_status.json",
            {
                "status": "complete",
                "shard_id": int(shard_id),
                "planned_cells": len(selected),
                "completed_cells": len(result_rows),
                "failed_cells": 0,
                "wall_time_total_s": wall_time,
                "heartbeat_at": time.time(),
            },
        )
    except Exception as error:
        _write_json(
            out / "shard_status.json",
            {
                "status": "failed",
                "shard_id": int(shard_id),
                "planned_cells": len(selected),
                "completed_cells": len(result_rows),
                "failed_cells": 1,
                "error": f"{type(error).__name__}: {error}",
                "heartbeat_at": time.time(),
            },
        )
        raise
    return out


def _summary(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        record: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        for metric in ("bands_2_10_mre_pct", "mae", "rmse"):
            numbers = [float(row[metric]) for row in values]
            record[f"{metric}_mean"] = statistics.mean(numbers)
            record[f"{metric}_std"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        record["cell_count"] = len(values)
        output.append(record)
    return output


def _bootstrap_pair(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if any(a.shape != b.shape for a, b in zip(left, right, strict=True)):
        raise ValueError("paired arrays do not have identical shapes")
    count = int(left[0].size)
    rng = np.random.default_rng(seed)
    distributions = {"mre": [], "mae": [], "rmse": []}
    remaining = int(replicates)
    while remaining:
        chunk = min(50, remaining)
        indices = rng.integers(0, count, size=(chunk, count))
        distributions["mre"].extend(np.mean(right[0][indices], axis=1) - np.mean(left[0][indices], axis=1))
        distributions["mae"].extend(np.mean(right[1][indices], axis=1) - np.mean(left[1][indices], axis=1))
        distributions["rmse"].extend(
            np.sqrt(np.mean(right[2][indices], axis=1)) - np.sqrt(np.mean(left[2][indices], axis=1))
        )
        remaining -= chunk
    result: dict[str, float] = {}
    points = {
        "mre": float(np.mean(right[0]) - np.mean(left[0])),
        "mae": float(np.mean(right[1]) - np.mean(left[1])),
        "rmse": float(np.sqrt(np.mean(right[2])) - np.sqrt(np.mean(left[2]))),
    }
    for metric, values in distributions.items():
        array = np.asarray(values, dtype=float)
        result[f"{metric}_delta"] = points[metric]
        result[f"{metric}_ci_low"] = float(np.quantile(array, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(array, 0.975))
    return result


def _plot_evaluation_summary(out: Path, current_summary: Sequence[Mapping[str, Any]], resolution_summary: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    rows = [row for row in current_summary if row["source_representation"] != row["target_representation"]]
    for model in MODELS:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: (row["source_representation"], int(row["epoch_budget"])),
        )
        axis.plot(
            range(len(selected)),
            [float(row["bands_2_10_mre_pct_mean"]) for row in selected],
            marker="o",
            label=model,
        )
    axis.set_ylabel("Cross-lane test MRE (%)")
    axis.set_xlabel("Direction and epoch-budget cells")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "current_40k_cross_evaluation.png", dpi=180)
    figure.savefig(figures / "current_40k_cross_evaluation.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for resolution in (32, 64, 128):
        values = [
            float(row["bands_2_10_mre_pct_mean"])
            for row in resolution_summary
            if int(row["evaluation_resolution"]) == resolution
        ]
        axis.boxplot(values, positions=[resolution], widths=10)
    axis.set_xticks([32, 64, 128])
    axis.set_xlabel("Input resolution")
    axis.set_ylabel("Test MRE across compatible frozen checkpoints (%)")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "zero_shot_resolution_transfer.png", dpi=180)
    figure.savefig(figures / "zero_shot_resolution_transfer.pdf")
    plt.close(figure)


def finalize_evaluation_campaign(
    *,
    campaign_dir: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    resolved = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    plan = _read_csv(campaign / "evaluation_plan.csv")
    inventory = _read_csv(campaign / "checkpoint_inventory.csv")
    _write_csv(
        campaign / "legacy_checkpoint_manifest.csv",
        [row for row in inventory if str(row["source_group"]).startswith("legacy10k")],
    )
    training_campaign = _read_json(_resolve(root, resolved["paths"]["training_campaign_status"]))
    _write_csv(
        campaign / "training_wave_manifest.csv",
        [
            {
                "index": wave["index"],
                "key": wave["key"],
                "representation": wave["representation"],
                "train_size": wave["train_size"],
                "max_epochs": wave["max_epochs"],
                "resolution": wave["resolution"],
                "status": wave["status"],
                "planned_runs": wave["planned_runs"],
                "manifest": wave["manifest"],
                "aggregation_dir": wave["aggregation_dir"],
                "job_ids": "|".join(str(value) for value in wave["job_ids"]),
            }
            for wave in training_campaign["waves"]
        ],
    )
    _write_csv(
        campaign / "cross_evaluation_manifest.csv",
        [row for row in plan if "cross" in str(row["contracts"])],
    )
    executable_plan = {row["evaluation_id"]: row for row in plan if row["planned_status"] == "planned"}
    completed_rows: list[dict[str, str]] = []
    sample_metrics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    resource_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for shard_id in range(int(resolved["shard_count"])):
        shard = campaign / "shards" / f"shard_{shard_id:02d}"
        status_path = shard / "shard_status.json"
        if not status_path.is_file():
            failures.append({"shard_id": shard_id, "error": "missing shard status"})
            continue
        status = _read_json(status_path)
        if status.get("status") != "complete":
            failures.append({"shard_id": shard_id, "error": status.get("error", status.get("status"))})
            continue
        rows = _read_csv(shard / "evaluation_cells.csv")
        completed_rows.extend(rows)
        with np.load(shard / "sample_metrics.npz", allow_pickle=False) as archive:
            ids = [str(value) for value in archive["evaluation_ids"]]
            shard_mre = np.asarray(archive["sample_mre_pct"])
            shard_mae = np.asarray(archive["sample_mae"])
            shard_mse = np.asarray(archive["sample_mse"])
            for index, evaluation_id in enumerate(ids):
                sample_metrics[evaluation_id] = (
                    shard_mre[index].copy(),
                    shard_mae[index].copy(),
                    shard_mse[index].copy(),
                )
        metrics = _read_json(shard / "metrics.json")
        meta = _read_json(shard / "run.meta.json")
        resource_rows.append({"shard_id": shard_id, **metrics, "slurm_job_id": meta.get("slurm_job_id"), "node": meta.get("slurm_node")})
    if failures:
        _write_csv(campaign / "failed_tasks.csv", failures)
        _write_json(
            campaign / "campaign_status.json",
            {
                "status": "evaluation_failed",
                "failed_shards": len(failures),
                "failed_cells": sum(int(row.get("failed_cells", 1)) for row in failures),
                "next_gate": "manual_failure_audit_no_automatic_retry",
            },
        )
        raise RuntimeError(f"evaluation shards are incomplete: {failures}")
    actual_ids = [row["evaluation_id"] for row in completed_rows]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(executable_plan):
        raise RuntimeError("completed evaluation IDs do not exactly match the frozen plan")
    if set(sample_metrics) != set(executable_plan):
        raise RuntimeError("sample metric IDs do not exactly match the frozen plan")

    all_rows: list[dict[str, Any]] = sorted(completed_rows, key=lambda row: int(row["plan_index"]))
    for row in plan:
        if row["planned_status"] != "planned":
            all_rows.append(
                {
                    **row,
                    "status": row["planned_status"],
                    "bands_2_10_mre_pct": "",
                    "mae": "",
                    "rmse": "",
                    "sample_count": 0,
                    "source_normalization_reused": True,
                    "fine_tuned": False,
                    "selection_influence": False,
                }
            )
    all_rows.sort(key=lambda row: int(row["plan_index"]))
    _write_csv(campaign / "evaluation_cells.csv", all_rows)
    _write_csv(campaign / "history.csv", all_rows)
    _write_csv(campaign / "resource_summary.csv", resource_rows)
    _write_csv(campaign / "failed_tasks.csv", [], fields=("shard_id", "error"))

    legacy_native = [row for row in completed_rows if "legacy10k_to_50k_native" in row["contracts"]]
    legacy_cross = [row for row in completed_rows if "legacy10k_to_50k_cross" in row["contracts"]]
    current = [row for row in completed_rows if row["source_group"] == "current40k" and int(row["evaluation_resolution"]) == 64]
    resolution = [row for row in completed_rows if "zero_shot_input_resolution_transfer" in row["contracts"]]
    legacy_native_summary = _summary(legacy_native, ("source_representation", "model", "training_resolution"))
    legacy_cross_summary = _summary(legacy_cross, ("source_representation", "target_representation", "model"))
    current_summary = _summary(current, ("source_representation", "target_representation", "model", "epoch_budget"))
    resolution_summary = _summary(
        resolution,
        ("source_group", "source_representation", "model", "epoch_budget", "evaluation_resolution"),
    )
    _write_csv(campaign / "legacy_native_50k_summary.csv", legacy_native_summary)
    _write_csv(campaign / "legacy_cross_evaluation_summary.csv", legacy_cross_summary)
    _write_csv(campaign / "current_40k_evaluation_summary.csv", current_summary)
    _write_csv(campaign / "resolution_transfer_summary.csv", resolution_summary)
    cross_cells = [row for row in completed_rows if "cross" in row["contracts"]]
    _write_csv(campaign / "cross_evaluation_cells.csv", cross_cells)
    seed_summary: list[dict[str, Any]] = []
    for analysis, rows in (
        ("legacy_native_50k", legacy_native_summary),
        ("legacy_cross_50k", legacy_cross_summary),
        ("current_40k_native_and_cross", current_summary),
        ("zero_shot_resolution", resolution_summary),
    ):
        seed_summary.extend({"analysis": analysis, **row} for row in rows)
    _write_csv(campaign / "seed_summary.csv", seed_summary)

    lookup: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in completed_rows:
        lookup[
            (
                row["source_group"],
                row["source_representation"],
                row["model"],
                row["seed"],
                row["epoch_budget"],
                f"{row['target_representation']}@{row['evaluation_resolution']}",
            )
        ] = row
    comparisons: list[dict[str, Any]] = []

    def compare(kind: str, left: Mapping[str, str], right: Mapping[str, str], label: str) -> None:
        seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
        comparisons.append(
            {
                "comparison_kind": kind,
                "comparison_id": label,
                "left_evaluation_id": left["evaluation_id"],
                "right_evaluation_id": right["evaluation_id"],
                "source_representation": right["source_representation"],
                "target_representation": right["target_representation"],
                "model": right["model"],
                "seed": right["seed"],
                **_bootstrap_pair(
                    sample_metrics[left["evaluation_id"]],
                    sample_metrics[right["evaluation_id"]],
                    replicates=int(resolved["bootstrap_replicates"]),
                    seed=seed,
                ),
            }
        )

    resolution_bundles: dict[tuple[str, str, str, str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in resolution:
        key = (row["source_group"], row["source_representation"], row["model"], row["seed"], row["epoch_budget"])
        resolution_bundles[key][int(row["evaluation_resolution"])] = row
    for key, values in sorted(resolution_bundles.items()):
        if set(values) != {32, 64, 128}:
            raise RuntimeError(f"resolution bundle is incomplete: {key}")
        for target_resolution in (32, 128):
            compare("zero_shot_resolution", values[64], values[target_resolution], f"resolution|{key}|64->{target_resolution}")

    for representation in ("binary", "continuous"):
        for model in MODELS:
            for seed_value in SEEDS:
                current_native = {
                    epoch: lookup[("current40k", representation, model, str(seed_value), str(epoch), f"{representation}@64")]
                    for epoch in (20, 100, 200)
                }
                for left_epoch, right_epoch in ((20, 100), (20, 200), (100, 200)):
                    compare(
                        "epoch_budget",
                        current_native[left_epoch],
                        current_native[right_epoch],
                        f"epoch|{representation}|{model}|seed{seed_value}|{left_epoch}->{right_epoch}",
                    )
                legacy = lookup[("legacy10k_main", representation, model, str(seed_value), "100", f"{representation}@64")]
                for epoch in (20, 100, 200):
                    compare(
                        "legacy10k_vs_current40k",
                        legacy,
                        current_native[epoch],
                        f"scale|{representation}|{model}|seed{seed_value}|10k100->40k{epoch}",
                    )
    _write_csv(campaign / "paired_comparisons.csv", comparisons)
    _plot_evaluation_summary(campaign, current_summary, resolution_summary)

    total_wall = sum(float(row["wall_time_total_s"]) for row in resource_rows)
    _write_json(
        campaign / "metrics.json",
        {
            "planned_executable_cells": EXPECTED_EXECUTABLE_CELLS,
            "completed_cells": len(completed_rows),
            "not_applicable_cells": EXPECTED_NOT_APPLICABLE_CELLS,
            "failed_cells": 0,
            "legacy_checkpoint_count": EXPECTED_LEGACY_MAIN + EXPECTED_LEGACY_ABLATION,
            "current_40k_checkpoint_count": EXPECTED_CURRENT_40K,
            "paired_comparison_count": len(comparisons),
            "bootstrap_replicates": int(resolved["bootstrap_replicates"]),
            "sum_shard_wall_time_s": total_wall,
            "inference_time_total_s": sum(float(row["inference_time_s"]) for row in completed_rows),
        },
    )
    _write_json(
        campaign / "results.json",
        {
            "status": "formal_50k_forward_evaluation_success",
            "completed_cells": len(completed_rows),
            "failed_cells": 0,
            "not_applicable_cells": EXPECTED_NOT_APPLICABLE_CELLS,
            "selection_influence": False,
            "source_normalization_reused": True,
            "current_50k_final_fit_test_executed": False,
            "current_50k_final_fit_boundary": "excluded_no_unbiased_external_test",
            "resolution_claim": "zero_shot_input_resolution_transfer_only",
            "next_gate": "result_report_and_documentation_audit",
        },
    )
    _write_json(
        campaign / "run.meta.json",
        {
            "script": "python -m phc2d.forward.full_cell_50k_evaluation finalize",
            "deployed_git_commit": os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
            "plan_hash": resolved["plan_hash"],
            "shard_count": int(resolved["shard_count"]),
        },
    )
    _write_json(
        campaign / "campaign_status.json",
        {
            "status": "evaluation_complete_pending_report",
            "plan_hash": resolved["plan_hash"],
            "shard_count": int(resolved["shard_count"]),
            "planned_cells": EXPECTED_EXECUTABLE_CELLS,
            "completed_cells": len(completed_rows),
            "not_applicable_cells": EXPECTED_NOT_APPLICABLE_CELLS,
            "failed_cells": 0,
            "next_gate": "result_report_and_documentation_audit",
        },
    )
    (campaign / "run.log").write_text(
        "\n".join(
            [
                "status=formal_50k_forward_evaluation_success",
                f"plan_hash={resolved['plan_hash']}",
                f"completed_cells={len(completed_rows)}",
                f"not_applicable_cells={EXPECTED_NOT_APPLICABLE_CELLS}",
                "failed_cells=0",
                f"paired_comparisons={len(comparisons)}",
                "next_gate=result_report_and_documentation_audit",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (campaign / "campaign_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": resolved["schema_version"],
                "status": "evaluation_complete_pending_report",
                "plan_hash": resolved["plan_hash"],
                "training_campaign_status": resolved["paths"]["training_campaign_status"],
                "checkpoint_inventory": "checkpoint_inventory.csv",
                "legacy_checkpoint_manifest": "legacy_checkpoint_manifest.csv",
                "training_wave_manifest": "training_wave_manifest.csv",
                "evaluation_plan": "evaluation_plan.csv",
                "cross_evaluation_manifest": "cross_evaluation_manifest.csv",
                "evaluation_cells": "evaluation_cells.csv",
                "cross_evaluation_cells": "cross_evaluation_cells.csv",
                "seed_summary": "seed_summary.csv",
                "paired_comparisons": "paired_comparisons.csv",
                "resource_summary": "resource_summary.csv",
                "failed_tasks": "failed_tasks.csv",
                "scientific_boundary": resolved["scientific_boundary"],
                "next_gate": "result_report_and_documentation_audit",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return campaign


def record_submission(
    *, campaign_dir: str | Path, shard_job_id: str, finalizer_job_id: str, repository_root: str | Path = "."
) -> None:
    campaign = _resolve(Path(repository_root).resolve(), campaign_dir)
    status = _read_json(campaign / "campaign_status.json")
    if status.get("status") != "evaluation_plan_ready":
        raise RuntimeError(f"campaign is not ready for first submission: {status.get('status')}")
    status.update(
        {
            "status": "evaluation_submitted",
            "shard_job_id": str(shard_job_id),
            "finalizer_job_id": str(finalizer_job_id),
            "next_gate": "evaluation_shards_and_finalizer_terminal",
        }
    )
    _write_json(campaign / "campaign_status.json", status)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--campaign-dir", type=Path, required=True)
    plan.add_argument("--legacy-main-manifest", type=Path, required=True)
    plan.add_argument("--legacy-ablation-manifest", type=Path, required=True)
    plan.add_argument("--training-campaign-status", type=Path, required=True)
    plan.add_argument("--binary-dataset", type=Path, required=True)
    plan.add_argument("--graded-dataset", type=Path, required=True)
    plan.add_argument("--shard-count", type=int, default=8)
    plan.add_argument("--bootstrap-replicates", type=int, default=500)
    run = subparsers.add_parser("run-shard")
    run.add_argument("--campaign-dir", type=Path, required=True)
    run.add_argument("--shard-id", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--campaign-dir", type=Path, required=True)
    submission = subparsers.add_parser("record-submission")
    submission.add_argument("--campaign-dir", type=Path, required=True)
    submission.add_argument("--shard-job-id", required=True)
    submission.add_argument("--finalizer-job-id", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--campaign-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        print(
            plan_evaluation_campaign(
                campaign_dir=args.campaign_dir,
                legacy_main_manifest=args.legacy_main_manifest,
                legacy_ablation_manifest=args.legacy_ablation_manifest,
                training_campaign_status=args.training_campaign_status,
                binary_dataset=args.binary_dataset,
                graded_dataset=args.graded_dataset,
                shard_count=args.shard_count,
                bootstrap_replicates=args.bootstrap_replicates,
            )
        )
    elif args.command == "run-shard":
        print(run_evaluation_shard(campaign_dir=args.campaign_dir, shard_id=args.shard_id, device=args.device))
    elif args.command == "finalize":
        print(finalize_evaluation_campaign(campaign_dir=args.campaign_dir))
    elif args.command == "record-submission":
        record_submission(
            campaign_dir=args.campaign_dir,
            shard_job_id=args.shard_job_id,
            finalizer_job_id=args.finalizer_job_id,
        )
    elif args.command == "status":
        print(json.dumps(_read_json(args.campaign_dir / "campaign_status.json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "build_evaluation_cells",
    "finalize_evaluation_campaign",
    "plan_evaluation_campaign",
    "record_submission",
    "run_evaluation_shard",
]
