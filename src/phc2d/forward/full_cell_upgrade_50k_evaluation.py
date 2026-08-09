"""Evaluate frozen 50k Full-Wh upgrade checkpoints and promotion gates.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell import _FullCellH5Dataset, _forward_model
from phc2d.forward.full_cell_upgrade_test import DIRECT_BASELINES
from phc2d.models import build_model, get_model_spec
from phc2d.training import DeviceConfig, resolve_device


EXPECTED_UPGRADE_SOURCES = 27
EXPECTED_BASELINE_SOURCES = 24
EXPECTED_EXECUTABLE_CELLS = 126
EXPECTED_NOT_APPLICABLE_CELLS = 18
RESOLUTION_MODELS = {
    "mlp_mixer_band_v1",
    "convnext_band_v1",
    "tfno2d_band_v3",
    "dual_lane_band_operator_v1",
}
LANE_TO_REPRESENTATION = {"B": "binary", "G": "continuous"}
REPRESENTATION_TO_LANE = {value: key for key, value in LANE_TO_REPRESENTATION.items()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    if not fields:
        raise ValueError(f"cannot write a schema-less CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _finite_json(payload: Mapping[str, Any]) -> bool:
    return not any(token in json.dumps(dict(payload), allow_nan=True) for token in ("NaN", "Infinity"))


def _checkpoint_hash(result: Mapping[str, Any], checkpoint: Path) -> str:
    expected = str(result.get("checkpoint_sha256", ""))
    if len(expected) != 64:
        raise RuntimeError(f"training result lacks checkpoint_sha256: {checkpoint.parent}")
    if _sha256(checkpoint) != expected:
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    return expected


def _training_runtime(result: Mapping[str, Any], metrics: Mapping[str, Any]) -> float:
    for payload in (metrics, result):
        for key in ("train_time_s", "wall_time_total_s", "wall_time_s"):
            if key in payload:
                return float(payload[key])
    return float("nan")


def _last_history_epoch(path: Path) -> int:
    rows = _read_csv(path)
    if not rows or "epoch" not in rows[-1]:
        raise RuntimeError(f"training history lacks an epoch column: {path}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, epochs[-1] + 1)):
        raise RuntimeError(f"training history epochs are not contiguous: {path}")
    if not 100 <= epochs[-1] <= 200:
        raise RuntimeError(f"50k actual stop epoch must be in [100, 200], got {epochs[-1]}")
    return epochs[-1]


def _load_upgrade_sources(root: Path, campaign: Path) -> list[dict[str, Any]]:
    manifest = _read_csv(campaign / "task_manifest.csv")
    selected = [row for row in manifest if row["action"] == "RUN"]
    if len(selected) != EXPECTED_UPGRADE_SOURCES:
        raise RuntimeError(f"50k training campaign must contain 27 RUN rows, got {len(selected)}")
    sources: list[dict[str, Any]] = []
    for row in selected:
        run = _resolve(root, row["run_dir"])
        required = (
            "run.log",
            "run.meta.json",
            "config.resolved.yaml",
            "results.json",
            "metrics.json",
            "history.csv",
            "checkpoint.pt",
            "model_summary.json",
            "epoch100.pt",
            "n40_profile.json",
        )
        missing = [name for name in required if not (run / name).is_file() or not (run / name).stat().st_size]
        if missing:
            raise RuntimeError(f"incomplete 50k training task {row['task_id']}: {missing}")
        result = _read_json(run / "results.json")
        metrics = _read_json(run / "metrics.json")
        meta = _read_json(run / "run.meta.json")
        profile = _read_json(run / "n40_profile.json")
        expected_status = "dual_lane_training_success" if row["lane"] == "J" else "formal_training_success"
        if result.get("status") != expected_status or profile.get("status") != "profile_collected":
            raise RuntimeError(f"50k training task is not admissible: {row['task_id']}")
        if result.get("selection_split", "validation") != "validation" or bool(result.get("test_executed")):
            raise RuntimeError(f"50k checkpoint was not selected without test leakage: {row['task_id']}")
        if not _finite_json(result) or not _finite_json(metrics):
            raise RuntimeError(f"non-finite 50k training artifact: {row['task_id']}")
        gate = result.get("epoch_100_continue_gate")
        if not isinstance(gate, Mapping) or gate.get("decision") not in {"stop_at_100", "continue_to_max_200"}:
            raise RuntimeError(f"missing epoch-100 gate record: {row['task_id']}")
        checkpoint = run / "checkpoint.pt"
        digest = _checkpoint_hash(result, checkpoint)
        actual_stop_epoch = _last_history_epoch(run / "history.csv")
        if gate["decision"] == "stop_at_100" and actual_stop_epoch != 100:
            raise RuntimeError(f"epoch-100 stop decision/history mismatch: {row['task_id']}")
        source = {
            "source_id": f"upgrade_{row['task_id']}",
            "source_group": "upgrade50k",
            "model": row["model"],
            "head": row["head"],
            "source_lane": row["lane"],
            "seed": int(row["seed"]),
            "checkpoint_path": _relative(root, checkpoint),
            "checkpoint_sha256": digest,
            "source_run_dir": _relative(root, run),
            "actual_stop_epoch": actual_stop_epoch,
            "best_validation_epoch": int(result.get("best_epoch", result.get("best_validation_epoch", 0))),
            "train_time_s": _training_runtime(result, metrics),
            "training_peak_gpu_memory_mib": float(profile.get("memory_used_mib_max", float("nan"))),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "git_commit": str(meta.get("git_commit", "")),
            "epoch_100_decision": str(gate["decision"]),
        }
        sources.append(source)
    identities = {(row["model"], row["source_lane"], row["seed"]) for row in sources}
    if len(identities) != EXPECTED_UPGRADE_SOURCES:
        raise RuntimeError("upgrade 50k source identities are incomplete or duplicated")
    commits = {str(row["git_commit"]) for row in sources}
    if len(commits) != 1 or "" in commits:
        raise RuntimeError(f"upgrade 50k source commits are not homogeneous: {sorted(commits)}")
    return sources


def _load_baseline_sources(root: Path, inventory: Path) -> list[dict[str, Any]]:
    expected = set(DIRECT_BASELINES.values())
    selected = [
        row
        for row in _read_csv(inventory)
        if row.get("source_group") == "current40k"
        and row.get("model") in expected
        and int(row.get("epoch_budget", 0)) == 200
        and int(row.get("training_resolution", 0)) == 64
    ]
    if len(selected) != EXPECTED_BASELINE_SOURCES:
        raise RuntimeError(f"baseline inventory must resolve 24 current40k/e200 sources, got {len(selected)}")
    sources: list[dict[str, Any]] = []
    for row in selected:
        lane = REPRESENTATION_TO_LANE.get(str(row["source_representation"]))
        if lane is None:
            raise ValueError(f"unsupported baseline representation: {row['source_representation']}")
        checkpoint = _resolve(root, row["checkpoint_path"])
        if not checkpoint.is_file() or _sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"baseline checkpoint hash mismatch: {checkpoint}")
        sources.append(
            {
                "source_id": f"baseline_{row['model']}_{lane}_s{row['seed']}",
                "source_group": "baseline40k_e200",
                "model": row["model"],
                "head": "legacy",
                "source_lane": lane,
                "seed": int(row["seed"]),
                "checkpoint_path": _relative(root, checkpoint),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "source_run_dir": row["run_dir"],
                "actual_stop_epoch": 200,
                "best_validation_epoch": "",
                "train_time_s": "",
                "training_peak_gpu_memory_mib": "",
                "checkpoint_bytes": checkpoint.stat().st_size,
                "git_commit": "historical_frozen_source",
                "epoch_100_decision": "historical_e200_budget",
            }
        )
    identities = {(row["model"], row["source_lane"], row["seed"]) for row in sources}
    if len(identities) != EXPECTED_BASELINE_SOURCES:
        raise RuntimeError("baseline source identities are incomplete or duplicated")
    return sources


def build_evaluation_cells(sources: Sequence[Mapping[str, Any]], *, shard_count: int = 8) -> list[dict[str, Any]]:
    """Build the preregistered native, cross-lane, and resolution matrix."""

    if int(shard_count) <= 0:
        raise ValueError("shard_count must be positive")
    cells: list[dict[str, Any]] = []

    def add(source: Mapping[str, Any], target_lane: str, resolution: int, contract: str, status: str = "planned") -> None:
        cells.append(
            {
                **dict(source),
                "target_lane": target_lane,
                "evaluation_resolution": int(resolution),
                "contract": contract,
                "planned_status": status,
            }
        )

    for source in sources:
        group = str(source["source_group"])
        source_lane = str(source["source_lane"])
        model = str(source["model"])
        if group == "baseline40k_e200":
            add(source, source_lane, 64, "native_baseline")
            continue
        if source_lane == "J":
            for lane in ("B", "G"):
                add(source, lane, 64, "native_joint")
                for resolution in (32, 128):
                    add(source, lane, resolution, "zero_shot_input_resolution")
            for direction in ("B_to_G", "G_to_B"):
                add(source, direction[-1], 64, f"cross_lane_{direction}", "not_applicable_joint_trained_both_lanes")
            continue
        add(source, source_lane, 64, "native_upgrade")
        opposite = "G" if source_lane == "B" else "B"
        add(source, opposite, 64, f"cross_lane_{source_lane}_to_{opposite}")
        if model in RESOLUTION_MODELS:
            for resolution in (32, 128):
                add(source, source_lane, resolution, "zero_shot_input_resolution")
        else:
            for resolution in (32, 128):
                add(source, source_lane, resolution, "zero_shot_input_resolution", "not_applicable_fixed_flatten_shape")

    executable = [row for row in cells if row["planned_status"] == "planned"]
    not_applicable = [row for row in cells if row["planned_status"] != "planned"]
    if len(executable) != EXPECTED_EXECUTABLE_CELLS or len(not_applicable) != EXPECTED_NOT_APPLICABLE_CELLS:
        raise RuntimeError(
            f"evaluation matrix mismatch: executable={len(executable)}, not_applicable={len(not_applicable)}"
        )
    weights: dict[str, int] = defaultdict(int)
    for row in executable:
        weights[str(row["source_id"])] += 1
    loads = [0] * int(shard_count)
    source_shards: dict[str, int] = {}
    for source_id, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        shard = min(range(int(shard_count)), key=lambda index: (loads[index], index))
        source_shards[source_id] = shard
        loads[shard] += weight
    ordered = sorted(
        cells,
        key=lambda row: (
            row["source_group"],
            row["model"],
            row["source_lane"],
            int(row["seed"]),
            row["contract"],
            row["target_lane"],
            int(row["evaluation_resolution"]),
        ),
    )
    for index, row in enumerate(ordered):
        row["evaluation_index"] = index
        row["evaluation_id"] = "up50eval_" + hashlib.sha256(
            f"{row['source_id']}|{row['target_lane']}|{row['evaluation_resolution']}|{row['contract']}".encode()
        ).hexdigest()[:20]
        row["shard_id"] = source_shards.get(str(row["source_id"]), "")
    return ordered


def _dataset_contracts(root: Path, training: Path) -> dict[str, dict[str, Any]]:
    import h5py

    rows = _read_csv(training / "task_manifest.csv")
    contracts: dict[str, dict[str, Any]] = {}
    for lane, label in (("B", "binary"), ("G", "graded")):
        row = next(item for item in rows if item["action"] == "RUN" and item["lane"] == lane)
        resolved = yaml.safe_load(_resolve(root, row["run_dir"]).joinpath("config.resolved.yaml").read_text(encoding="utf-8"))
        dataset = dict(resolved["dataset"])
        path = _resolve(root, dataset["path"])
        with h5py.File(path, "r") as handle:
            idx_test = np.asarray(handle["idx_test"], dtype=np.int64)
            k_points = np.asarray(handle["k_points"], dtype=np.float32)
            fields = {int(value): f"material_weight_grid_{int(value)}" for value in (32, 64, 128)}
            missing = [field for field in fields.values() if field not in handle]
            if missing:
                raise KeyError(f"{label} dataset lacks zero-shot resolution fields: {missing}")
        expected_hash = str(dataset.get("expected_dataset_sha256", ""))
        actual_hash = _sha256(path)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"{label} dataset hash mismatch while freezing evaluation")
        contracts[lane] = {
            "label": label,
            "path": _relative(root, path),
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
            "idx_test_sha256": _sha256_array(idx_test),
            "test_count": int(idx_test.size),
            "k_points_sha256": _sha256_array(k_points),
            "grid_fields": fields,
            "sampling": "deterministic_threshold" if lane == "B" else "continuous",
            "threshold": 0.5 if lane == "B" else None,
        }
    if contracts["B"]["k_points_sha256"] != contracts["G"]["k_points_sha256"]:
        raise RuntimeError("binary and graded 50k k-point hashes differ")
    return contracts


def plan_evaluation_campaign(
    training_campaign: str | Path,
    *,
    baseline_inventory: str | Path,
    output_dir: str | Path,
    repository_root: str | Path = ".",
    shard_count: int = 8,
    bootstrap_replicates: int = 10_000,
) -> Path:
    """Freeze the post-W4 evaluation matrix without reading held-out metrics."""

    root = Path(repository_root).resolve()
    training = _resolve(root, training_campaign)
    baseline = _resolve(root, baseline_inventory)
    out = _resolve(root, output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"evaluation campaign is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sources = _load_upgrade_sources(root, training) + _load_baseline_sources(root, baseline)
    cells = build_evaluation_cells(sources, shard_count=int(shard_count))
    datasets = _dataset_contracts(root, training)
    _write_csv(out / "checkpoint_inventory.csv", sources)
    _write_csv(out / "evaluation_plan.csv", cells)
    plan_hash = hashlib.sha256((out / "evaluation_plan.csv").read_bytes()).hexdigest()
    config = {
        "schema_version": "phc2d.full_cell_upgrade_50k_evaluation.v1",
        "training_campaign": _relative(root, training),
        "training_manifest_sha256": _sha256(training / "task_manifest.csv"),
        "baseline_inventory": _relative(root, baseline),
        "baseline_inventory_sha256": _sha256(baseline),
        "plan_hash": plan_hash,
        "shard_count": int(shard_count),
        "bootstrap_replicates": int(bootstrap_replicates),
        "datasets": datasets,
        "expected_executable_cells": EXPECTED_EXECUTABLE_CELLS,
        "expected_not_applicable_cells": EXPECTED_NOT_APPLICABLE_CELLS,
        "selection_boundary": "all checkpoints and e200 baselines frozen before upgrade 50k test",
        "resolution_boundary": "zero_shot_input_resolution_only_not_physical_convergence",
        "cross_lane_boundary": "domain_shift_diagnostic_not_native_promotion_metric",
    }
    (out / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        out / "campaign_status.json",
        {
            "status": "evaluation_plan_ready",
            "plan_hash": plan_hash,
            "executable_cells": EXPECTED_EXECUTABLE_CELLS,
            "not_applicable_cells": EXPECTED_NOT_APPLICABLE_CELLS,
            "test_metrics_read": False,
            "next_gate": "authorized_n40_evaluation_submission",
        },
    )
    return out


def _checkpoint_payload(path: Path, *, device: Any, torch: Any) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {path}")
    return payload


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _source_normalizer(payload: Mapping[str, Any], *, source_lane: str, target_lane: str) -> tuple[np.ndarray, np.ndarray]:
    if source_lane == "J":
        lane_name = "binary" if target_lane == "B" else "graded"
        normalizer = dict(payload["target_normalizers"])[lane_name]
        return _as_numpy(normalizer["mean"]), _as_numpy(normalizer["std"])
    return _as_numpy(payload["target_mean"]), _as_numpy(payload["target_std"])


def _evaluate_cell(
    *,
    model: Any,
    model_spec: Any,
    payload: Mapping[str, Any],
    row: Mapping[str, str],
    dataset_contract: Mapping[str, Any],
    root: Path,
    device_info: Any,
    torch: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    import h5py
    from torch.utils.data import DataLoader

    target_lane = str(row["target_lane"])
    source_lane = str(row["source_lane"])
    resolution = int(row["evaluation_resolution"])
    path = _resolve(root, dataset_contract["path"])
    if path.stat().st_size != int(dataset_contract["size_bytes"]):
        raise RuntimeError(f"dataset size changed after evaluation freeze: {path}")
    with h5py.File(path, "r") as handle:
        indices = np.asarray(handle["idx_test"], dtype=np.int64)
        k_points_np = np.asarray(handle["k_points"], dtype=np.float32)
    if _sha256_array(indices) != dataset_contract["idx_test_sha256"]:
        raise RuntimeError(f"test split hash changed for lane {target_lane}")
    if _sha256_array(k_points_np) != dataset_contract["k_points_sha256"]:
        raise RuntimeError(f"k-point hash changed for lane {target_lane}")
    mean_np, std_np = _source_normalizer(payload, source_lane=source_lane, target_lane=target_lane)
    dataset = _FullCellH5Dataset(
        path=path,
        indices=indices,
        grid_field=str(dict(dataset_contract["grid_fields"])[resolution]),
        layout=str(model_spec.input_layout),
        sampling=str(dataset_contract["sampling"]),
        threshold=dataset_contract["threshold"],
        target_mean=mean_np,
        target_std=std_np,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=int(device_info.num_workers),
        pin_memory=bool(device_info.pin_memory),
    )
    k_points = (
        torch.as_tensor(k_points_np, dtype=torch.float32, device=device_info.torch_device)
        if model_spec.accepts_k_points
        else None
    )
    lane_index = None if source_lane != "J" else (0 if target_lane == "B" else 1)
    model.eval()
    squared = 0.0
    absolute = 0.0
    relative_sum = 0.0
    count = 0
    sample_offset = 0
    per_band_sum: np.ndarray | None = None
    per_band_count = 0
    sample_mre: list[np.ndarray] = []
    sample_full_mre: list[np.ndarray] = []
    sample_ordering: list[np.ndarray] = []
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device_info.torch_device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device_info.torch_device)
    if device_info.is_cuda:
        torch.cuda.synchronize(device_info.torch_device)
    started = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device_info.torch_device, non_blocking=True)
            yb = yb.to(device_info.torch_device, non_blocking=True)
            prediction_norm = _forward_model(
                model,
                xb,
                model_spec=model_spec,
                k_points=k_points,
                lane_index=lane_index,
                torch=torch,
            )
            prediction = prediction_norm * std + mean
            target = yb * std + mean
            pred_np = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
            target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
            diff = pred_np - target_np
            relative = np.abs(diff) / np.maximum(np.abs(target_np), 1.0e-8)
            squared += float(np.sum(diff * diff))
            absolute += float(np.sum(np.abs(diff)))
            relative_sum += float(np.sum(relative))
            count += int(diff.size)
            band_values = np.sum(relative, axis=(0, 1))
            per_band_sum = band_values if per_band_sum is None else per_band_sum + band_values
            per_band_count += int(diff.shape[0] * diff.shape[1])
            sample_mre.append(100.0 * np.mean(relative[..., 1:], axis=(1, 2)))
            sample_full_mre.append(100.0 * np.mean(relative, axis=(1, 2)))
            sample_ordering.append(np.mean(np.diff(pred_np, axis=-1) < 0.0, axis=(1, 2)))
            sample_offset += int(diff.shape[0])
    if device_info.is_cuda:
        torch.cuda.synchronize(device_info.torch_device)
    elapsed = time.perf_counter() - started
    if sample_offset != int(dataset_contract["test_count"]) or per_band_sum is None:
        raise RuntimeError(f"test sample count mismatch for {row['evaluation_id']}")
    sample_values = np.concatenate(sample_mre).astype(np.float64, copy=False)
    full_values = np.concatenate(sample_full_mre).astype(np.float64, copy=False)
    ordering_values = np.concatenate(sample_ordering).astype(np.float64, copy=False)
    metrics = {
        "test_bands_2_10_mre_pct": float(np.mean(sample_values)),
        "test_full_matrix_mre_pct": float(np.mean(full_values)),
        "test_mae": absolute / max(count, 1),
        "test_rmse": math.sqrt(squared / max(count, 1)),
        "test_ordering_violation_rate": float(np.mean(ordering_values)),
        "n_test": int(sample_values.size),
        "eval_time_s": elapsed,
        "inference_time_per_sample_ms": 1000.0 * elapsed / max(sample_values.size, 1),
        "throughput_samples_per_s": sample_values.size / max(elapsed, 1.0e-12),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError(f"non-finite evaluation metrics: {row['evaluation_id']}")
    per_band = [
        {
            "evaluation_id": row["evaluation_id"],
            "band_one_based": index + 1,
            "mre_pct": float(100.0 * value / per_band_count),
        }
        for index, value in enumerate(per_band_sum)
    ]
    return metrics, per_band, indices, sample_values


def run_evaluation_shard(
    campaign_dir: str | Path,
    *,
    shard_id: int,
    repository_root: str | Path = ".",
) -> Path:
    """Evaluate all frozen cells assigned to one single-GPU shard."""

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    config = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    plan = _read_csv(campaign / "evaluation_plan.csv")
    selected = [
        row for row in plan
        if row["planned_status"] == "planned" and int(row["shard_id"]) == int(shard_id)
    ]
    if not selected:
        raise RuntimeError(f"evaluation shard {shard_id} has no planned cells")
    out = campaign / "shards" / f"shard_{int(shard_id):02d}"
    out.mkdir(parents=True, exist_ok=False)
    import torch

    device_info = resolve_device(
        DeviceConfig(
            device="cuda:0",
            require_accelerator=True,
            precision="fp32",
            amp_enabled=False,
            tf32_enabled=False,
            num_workers=4,
            pin_memory=True,
        ),
        torch_module=torch,
    )
    torch.cuda.reset_peak_memory_stats(device_info.torch_device)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[row["source_id"]].append(row)
    result_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    evaluation_ids: list[str] = []
    dataset_indices: list[np.ndarray] = []
    sample_mre: list[np.ndarray] = []
    started = time.perf_counter()
    for source_id in sorted(grouped):
        rows = sorted(grouped[source_id], key=lambda row: int(row["evaluation_index"]))
        first = rows[0]
        checkpoint = _resolve(root, first["checkpoint_path"])
        if _sha256(checkpoint) != first["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed after evaluation freeze: {checkpoint}")
        payload = _checkpoint_payload(checkpoint, device=device_info.torch_device, torch=torch)
        if str(payload.get("model_name")) != first["model"]:
            raise RuntimeError(f"checkpoint model identity mismatch: {source_id}")
        spec = get_model_spec(first["model"])
        model_kwargs = dict(payload["model_params"])
        if spec.accepts_k_points:
            model_kwargs["k_points"] = payload.get("k_points")
        model = build_model(
            spec.name,
            input_shape=tuple(payload["input_shape"]),
            output_shape=tuple(payload["output_shape"]),
            **model_kwargs,
        ).to(device_info.torch_device)
        model.load_state_dict(payload["state_dict"])
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        for row in rows:
            metrics, per_band, indices, values = _evaluate_cell(
                model=model,
                model_spec=spec,
                payload=payload,
                row=row,
                dataset_contract=dict(config["datasets"])[row["target_lane"]],
                root=root,
                device_info=device_info,
                torch=torch,
            )
            result_rows.append(
                {
                    **row,
                    **metrics,
                    "status": "completed",
                    "fine_tuned": False,
                    "parameter_count": parameter_count,
                }
            )
            band_rows.extend(per_band)
            evaluation_ids.append(row["evaluation_id"])
            dataset_indices.append(indices)
            sample_mre.append(values)
            _write_csv(out / "history.csv", result_rows)
    _write_csv(out / "evaluation_cells.csv", result_rows)
    _write_csv(out / "per_band_metrics.csv", band_rows)
    with (out / "sample_metrics.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            evaluation_ids=np.asarray(evaluation_ids),
            dataset_indices=np.stack(dataset_indices),
            sample_mre_pct=np.stack(sample_mre),
        )
    wall = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(device_info.torch_device) / (1024**2))
    _write_json(
        out / "metrics.json",
        {
            "completed_cells": len(result_rows),
            "wall_time_total_s": wall,
            "eval_time_total_s": sum(float(row["eval_time_s"]) for row in result_rows),
            "peak_gpu_memory_mb": peak,
        },
    )
    _write_json(
        out / "results.json",
        {
            "status": "formal_upgrade_50k_evaluation_shard_success",
            "shard_id": int(shard_id),
            "completed_cells": len(result_rows),
            "plan_hash": config["plan_hash"],
        },
    )
    _write_json(
        out / "run.meta.json",
        {
            "script": "python -m phc2d.forward.full_cell_upgrade_50k_evaluation run-shard",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            **device_info.metadata(),
        },
    )
    (out / "config.resolved.yaml").write_text(
        yaml.safe_dump({"campaign": _relative(root, campaign), "shard_id": int(shard_id)}, sort_keys=False),
        encoding="utf-8",
    )
    (out / "run.log").write_text("frozen upgrade 50k evaluation shard completed.\n", encoding="utf-8")
    return out


def _summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        members = sorted(members, key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in members]
        if seeds != [0, 1, 2]:
            raise RuntimeError(f"summary group {key} lacks exactly seeds 0,1,2: {seeds}")
        values = [float(row["test_bands_2_10_mre_pct"]) for row in members]
        record: dict[str, Any] = dict(zip(fields, key, strict=True))
        record.update(
            {
                "seed_0": values[0],
                "seed_1": values[1],
                "seed_2": values[2],
                "mean_mre_pct": statistics.fmean(values),
                "std_mre_pct": statistics.stdev(values),
                "worst_seed_mre_pct": max(values),
                "mean_full_matrix_mre_pct": statistics.fmean(
                    float(row["test_full_matrix_mre_pct"]) for row in members
                ),
                "mean_mae": statistics.fmean(float(row["test_mae"]) for row in members),
                "mean_rmse": statistics.fmean(float(row["test_rmse"]) for row in members),
                "mean_ordering_violation_rate": statistics.fmean(
                    float(row["test_ordering_violation_rate"]) for row in members
                ),
                "parameter_count": int(members[0]["parameter_count"]),
                "checkpoint_bytes_mean": statistics.fmean(float(row["checkpoint_bytes"]) for row in members),
                "actual_stop_epoch_0": int(members[0]["actual_stop_epoch"]),
                "actual_stop_epoch_1": int(members[1]["actual_stop_epoch"]),
                "actual_stop_epoch_2": int(members[2]["actual_stop_epoch"]),
                "best_validation_epoch_0": members[0]["best_validation_epoch"],
                "best_validation_epoch_1": members[1]["best_validation_epoch"],
                "best_validation_epoch_2": members[2]["best_validation_epoch"],
                "epoch_100_decision_0": members[0]["epoch_100_decision"],
                "epoch_100_decision_1": members[1]["epoch_100_decision"],
                "epoch_100_decision_2": members[2]["epoch_100_decision"],
                "train_time_s_mean": statistics.fmean(float(row["train_time_s"]) for row in members)
                if all(str(row["train_time_s"]) not in {"", "nan"} for row in members)
                else "",
                "training_peak_gpu_memory_mib_max": max(
                    float(row["training_peak_gpu_memory_mib"]) for row in members
                )
                if all(str(row["training_peak_gpu_memory_mib"]) not in {"", "nan"} for row in members)
                else "",
                "eval_time_s_mean": statistics.fmean(float(row["eval_time_s"]) for row in members),
                "inference_time_per_sample_ms_mean": statistics.fmean(
                    float(row["inference_time_per_sample_ms"]) for row in members
                ),
                "throughput_samples_per_s_mean": statistics.fmean(
                    float(row["throughput_samples_per_s"]) for row in members
                ),
                "evaluation_peak_gpu_memory_mb_max": max(
                    float(row["evaluation_peak_gpu_memory_mb"]) for row in members
                ),
            }
        )
        output.append(record)
    return output


def _paired_bootstrap(
    delta: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float, float]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap requires a finite non-empty vector")
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(replicates), dtype=np.float64)
    chunk = 200
    for start in range(0, int(replicates), chunk):
        stop = min(start + chunk, int(replicates))
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        estimates[start:stop] = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _plot_summaries(
    out: Path,
    native: Sequence[Mapping[str, Any]],
    cross: Sequence[Mapping[str, Any]],
    resolution: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def bar_plot(rows: Sequence[Mapping[str, Any]], *, title: str, name: str, label_key: Any) -> None:
        labels = [label_key(row) for row in rows]
        values = [float(row["mean_mre_pct"]) for row in rows]
        figure, axis = plt.subplots(figsize=(max(9.0, 0.45 * len(rows)), 5.2))
        axis.bar(range(len(values)), values, color="#3B6FB6")
        axis.set_xticks(range(len(values)), labels, rotation=55, ha="right", fontsize=8)
        axis.set_ylabel("Bands 2–10 MRE (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(figures / f"{name}.png", dpi=180)
        figure.savefig(figures / f"{name}.pdf")
        plt.close(figure)

    bar_plot(
        native,
        title="Frozen 50k held-out native evaluation",
        name="native_test_comparison",
        label_key=lambda row: f"{row['model']}:{row['target_lane']}",
    )
    bar_plot(
        cross,
        title="Cross-lane domain-shift diagnostic",
        name="cross_lane_diagnostic",
        label_key=lambda row: f"{row['model']}:{row['source_lane']}→{row['target_lane']}",
    )
    bar_plot(
        resolution,
        title="Zero-shot input-resolution diagnostic",
        name="zero_shot_resolution_diagnostic",
        label_key=lambda row: f"{row['model']}:{row['target_lane']}@{row['evaluation_resolution']}",
    )


def _native_per_band_summary(
    band_rows: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cells = {str(row["evaluation_id"]): row for row in completed}
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in band_rows:
        cell = cells[str(row["evaluation_id"])]
        if cell["source_group"] != "upgrade50k" or not str(cell["contract"]).startswith("native"):
            continue
        key = (
            str(cell["model"]),
            str(cell["head"]),
            str(cell["target_lane"]),
            int(row["band_one_based"]),
        )
        grouped[key].append(float(row["mre_pct"]))
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        if len(values) != 3:
            raise RuntimeError(f"native per-band group lacks three seeds: {key}")
        output.append(
            {
                "model": key[0],
                "head": key[1],
                "lane": key[2],
                "band_one_based": key[3],
                "mean_mre_pct": statistics.fmean(values),
                "std_mre_pct": statistics.stdev(values),
            }
        )
    if len(output) != 100:
        raise RuntimeError(f"native upgrade per-band summary must contain 100 rows, got {len(output)}")
    return output


def finalize_evaluation_campaign(
    campaign_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Aggregate all shards and apply the preregistered promotion rules."""

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    config = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    plan = _read_csv(campaign / "evaluation_plan.csv")
    planned = [row for row in plan if row["planned_status"] == "planned"]
    not_applicable = [row for row in plan if row["planned_status"] != "planned"]
    if len(planned) != EXPECTED_EXECUTABLE_CELLS or len(not_applicable) != EXPECTED_NOT_APPLICABLE_CELLS:
        raise RuntimeError("evaluation plan identity changed before finalization")
    completed: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    sample_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shard_metrics: list[dict[str, Any]] = []
    for shard_id in range(int(config["shard_count"])):
        shard = campaign / "shards" / f"shard_{shard_id:02d}"
        required = (
            "run.log",
            "run.meta.json",
            "config.resolved.yaml",
            "results.json",
            "metrics.json",
            "history.csv",
            "evaluation_cells.csv",
            "per_band_metrics.csv",
            "sample_metrics.npz",
            "n40_profile.csv",
            "n40_profile.json",
        )
        missing = [name for name in required if not (shard / name).is_file() or not (shard / name).stat().st_size]
        if missing:
            raise RuntimeError(f"evaluation shard {shard_id} is incomplete: {missing}")
        result = _read_json(shard / "results.json")
        profile = _read_json(shard / "n40_profile.json")
        metrics = _read_json(shard / "metrics.json")
        if result.get("status") != "formal_upgrade_50k_evaluation_shard_success":
            raise RuntimeError(f"evaluation shard {shard_id} did not succeed")
        if result.get("plan_hash") != config["plan_hash"] or profile.get("status") != "profile_collected":
            raise RuntimeError(f"evaluation shard {shard_id} identity/profile mismatch")
        shard_rows = _read_csv(shard / "evaluation_cells.csv")
        for row in shard_rows:
            row["evaluation_peak_gpu_memory_mb"] = float(metrics["peak_gpu_memory_mb"])
        completed.extend(shard_rows)
        band_rows.extend(_read_csv(shard / "per_band_metrics.csv"))
        shard_metrics.append({"shard_id": shard_id, **metrics, **profile})
        with np.load(shard / "sample_metrics.npz", allow_pickle=False) as payload:
            ids = payload["evaluation_ids"].astype(str).tolist()
            indices = payload["dataset_indices"]
            values = payload["sample_mre_pct"]
        for index, evaluation_id in enumerate(ids):
            if evaluation_id in sample_lookup:
                raise RuntimeError(f"duplicate sample metrics: {evaluation_id}")
            sample_lookup[evaluation_id] = (indices[index].astype(np.int64), values[index].astype(np.float64))
    expected_ids = {row["evaluation_id"] for row in planned}
    actual_ids = {row["evaluation_id"] for row in completed}
    if len(completed) != EXPECTED_EXECUTABLE_CELLS or actual_ids != expected_ids or set(sample_lookup) != expected_ids:
        raise RuntimeError("evaluation shards do not exactly cover the frozen plan")
    _write_csv(campaign / "evaluation_cells.csv", completed)
    _write_csv(campaign / "per_band_metrics.csv", band_rows)
    _write_csv(campaign / "native_per_band_summary.csv", _native_per_band_summary(band_rows, completed))
    _write_csv(campaign / "not_applicable_cells.csv", not_applicable)
    native_rows = [row for row in completed if row["contract"].startswith("native")]
    cross_rows = [row for row in completed if row["contract"].startswith("cross_lane")]
    resolution_rows = [row for row in completed if row["contract"] == "zero_shot_input_resolution"]
    native_summary = _summary(native_rows, ("source_group", "model", "head", "source_lane", "target_lane"))
    cross_summary = _summary(cross_rows, ("model", "head", "source_lane", "target_lane"))
    resolution_summary = _summary(
        resolution_rows,
        ("model", "head", "source_lane", "target_lane", "evaluation_resolution"),
    )
    _write_csv(campaign / "native_test_summary.csv", native_summary)
    _write_csv(campaign / "cross_lane_summary.csv", cross_summary)
    _write_csv(campaign / "resolution_transfer_summary.csv", resolution_summary)
    native_lookup = {
        (row["source_group"], row["model"], row["target_lane"]): row for row in native_summary
    }

    grouped_samples: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    grouped_native: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in native_rows:
        grouped_native[(row["source_group"], row["model"], row["target_lane"])].append(row)
    for key, rows in grouped_native.items():
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        arrays = [sample_lookup[row["evaluation_id"]] for row in rows]
        if len(arrays) != 3 or any(not np.array_equal(arrays[0][0], item[0]) for item in arrays[1:]):
            raise RuntimeError(f"paired seed sample contract mismatch: {key}")
        grouped_samples[key] = (arrays[0][0], np.mean(np.stack([item[1] for item in arrays]), axis=0))

    comparisons: list[dict[str, Any]] = []

    def compare(
        *,
        comparison: str,
        model: str,
        lane: str,
        reference_group: str,
        reference_model: str,
        seed: int,
    ) -> dict[str, Any]:
        upgrade_key = ("upgrade50k", model, lane)
        reference_key = (reference_group, reference_model, lane)
        upgrade_indices, upgrade_values = grouped_samples[upgrade_key]
        reference_indices, reference_values = grouped_samples[reference_key]
        if not np.array_equal(upgrade_indices, reference_indices):
            raise RuntimeError(f"paired test indices differ: {comparison}/{model}/{lane}")
        estimate, low, high = _paired_bootstrap(
            upgrade_values - reference_values,
            seed=seed,
            replicates=int(config["bootstrap_replicates"]),
        )
        upgrade_mean = float(native_lookup[upgrade_key]["mean_mre_pct"])
        reference_mean = float(native_lookup[reference_key]["mean_mre_pct"])
        record = {
            "comparison": comparison,
            "model": model,
            "lane": lane,
            "reference_group": reference_group,
            "reference_model": reference_model,
            "upgrade_mean_mre_pct": upgrade_mean,
            "reference_mean_mre_pct": reference_mean,
            "relative_improvement": (reference_mean - upgrade_mean) / reference_mean,
            "upgrade_minus_reference_mre_pct_points": estimate,
            "ci95_low": low,
            "ci95_high": high,
            "paired_sample_count": int(upgrade_values.size),
            "bootstrap_replicates": int(config["bootstrap_replicates"]),
        }
        comparisons.append(record)
        return record

    direct_comparisons: dict[tuple[str, str], dict[str, Any]] = {}
    counter = 0
    for model, baseline_model in DIRECT_BASELINES.items():
        if model == "dual_lane_band_operator_v1":
            continue
        for lane in ("B", "G"):
            direct_comparisons[(model, lane)] = compare(
                comparison="direct_baseline",
                model=model,
                lane=lane,
                reference_group="baseline40k_e200",
                reference_model=baseline_model,
                seed=20260809 + counter,
            )
            counter += 1
    fno_comparisons: dict[tuple[str, str], dict[str, Any]] = {}
    for model in DIRECT_BASELINES:
        for lane in ("B", "G"):
            fno_comparisons[(model, lane)] = compare(
                comparison="current_best_fno_e200",
                model=model,
                lane=lane,
                reference_group="baseline40k_e200",
                reference_model="fno2d_band_v2",
                seed=20260809 + counter,
            )
            counter += 1
    independent_models = tuple(model for model in DIRECT_BASELINES if model != "dual_lane_band_operator_v1")
    best_independent: dict[str, str] = {}
    joint_comparisons: dict[str, dict[str, Any]] = {}
    for lane in ("B", "G"):
        best_model = min(
            independent_models,
            key=lambda model: float(native_lookup[("upgrade50k", model, lane)]["mean_mre_pct"]),
        )
        best_independent[lane] = best_model
        joint_comparisons[lane] = compare(
            comparison="joint_vs_best_independent_upgrade",
            model="dual_lane_band_operator_v1",
            lane=lane,
            reference_group="upgrade50k",
            reference_model=best_model,
            seed=20260809 + counter,
        )
        counter += 1
    _write_csv(campaign / "paired_bootstrap_comparisons.csv", comparisons)

    promotion: list[dict[str, Any]] = []
    for model in independent_models:
        lanes = [direct_comparisons[(model, lane)] for lane in ("B", "G")]
        improvements = [float(row["relative_improvement"]) for row in lanes]
        significant_ten = [
            value >= 0.10 and float(row["ci95_high"]) < 0.0
            for value, row in zip(improvements, lanes, strict=True)
        ]
        parameter_count = int(native_lookup[("upgrade50k", model, "B")]["parameter_count"])
        artifacts_ok = 2_000_000 <= parameter_count <= 8_000_000
        promoted = any(significant_ten) and min(improvements) >= -0.03 and artifacts_ok
        if promoted:
            status = "promoted_over_direct_baseline"
        elif min(improvements) < -0.03:
            status = "regressed"
        elif max(improvements) > 0.0:
            status = "local_improvement_only"
        else:
            status = "no_clear_gain"
        promotion.append(
            {
                "model": model,
                "status": status,
                "parameter_count": parameter_count,
                "parameter_budget_met": artifacts_ok,
                "binary_relative_improvement": improvements[0],
                "graded_relative_improvement": improvements[1],
                "binary_ci95_high": lanes[0]["ci95_high"],
                "graded_ci95_high": lanes[1]["ci95_high"],
            }
        )
    overall_fno_models = []
    for model in DIRECT_BASELINES:
        rows = [fno_comparisons[(model, lane)] for lane in ("B", "G")]
        if all(float(row["relative_improvement"]) >= 0.05 and float(row["ci95_high"]) < 0.0 for row in rows):
            overall_fno_models.append(model)
    joint_native = {lane: native_lookup[("upgrade50k", "dual_lane_band_operator_v1", lane)] for lane in ("B", "G")}
    joint_degradation = {
        lane: -float(joint_comparisons[lane]["relative_improvement"]) for lane in ("B", "G")
    }
    joint_worst_regression = {
        lane: (
            float(joint_native[lane]["worst_seed_mre_pct"])
            - float(native_lookup[("upgrade50k", best_independent[lane], lane)]["worst_seed_mre_pct"])
        )
        / float(native_lookup[("upgrade50k", best_independent[lane], lane)]["worst_seed_mre_pct"])
        for lane in ("B", "G")
    }
    joint_params = int(joint_native["B"]["parameter_count"])
    independent_params = sum(
        int(native_lookup[("upgrade50k", best_independent[lane], lane)]["parameter_count"])
        for lane in ("B", "G")
    )
    joint_checkpoint_bytes = float(joint_native["B"]["checkpoint_bytes_mean"])
    independent_checkpoint_bytes = sum(
        float(native_lookup[("upgrade50k", best_independent[lane], lane)]["checkpoint_bytes_mean"])
        for lane in ("B", "G")
    )
    checkpoint_storage_fraction = joint_checkpoint_bytes / independent_checkpoint_bytes
    joint_replace = (
        all(value <= 0.03 for value in joint_degradation.values())
        and joint_params <= 0.60 * independent_params
        and checkpoint_storage_fraction <= 0.60
        and all(value <= 0.05 for value in joint_worst_regression.values())
    )
    promotion.append(
        {
            "model": "dual_lane_band_operator_v1",
            "status": "joint_replacement_promoted" if joint_replace else "joint_replacement_not_met",
            "parameter_count": joint_params,
            "independent_parameter_count_sum": independent_params,
            "parameter_fraction": joint_params / independent_params,
            "joint_checkpoint_bytes_mean": joint_checkpoint_bytes,
            "independent_checkpoint_bytes_mean_sum": independent_checkpoint_bytes,
            "checkpoint_storage_fraction": checkpoint_storage_fraction,
            "best_binary_independent": best_independent["B"],
            "best_graded_independent": best_independent["G"],
            "binary_relative_degradation": joint_degradation["B"],
            "graded_relative_degradation": joint_degradation["G"],
            "binary_worst_seed_relative_regression": joint_worst_regression["B"],
            "graded_worst_seed_relative_regression": joint_worst_regression["G"],
        }
    )
    _write_csv(campaign / "promotion_decisions.csv", promotion)
    _write_csv(campaign / "resource_summary.csv", shard_metrics)
    _plot_summaries(campaign, native_summary, cross_summary, resolution_summary)
    total_wall = sum(float(row["wall_time_total_s"]) for row in shard_metrics)
    _write_json(
        campaign / "metrics.json",
        {
            "native_evaluation_rows": len(native_rows),
            "cross_lane_evaluation_rows": len(cross_rows),
            "resolution_evaluation_rows": len(resolution_rows),
            "not_applicable_rows": len(not_applicable),
            "bootstrap_comparisons": len(comparisons),
            "bootstrap_replicates": int(config["bootstrap_replicates"]),
            "sum_shard_wall_time_s": total_wall,
        },
    )
    _write_json(
        campaign / "results.json",
        {
            "status": "formal_upgrade_50k_evaluation_success",
            "formal_completed": True,
            "promoted_over_direct_baseline": [
                row["model"] for row in promotion if row["status"] == "promoted_over_direct_baseline"
            ],
            "overall_better_than_current_fno_models": overall_fno_models,
            "overall_better_than_current_fno": bool(overall_fno_models),
            "joint_replacement_status": promotion[-1]["status"],
            "cross_lane_evidence_role": "domain_shift_diagnostic",
            "resolution_evidence_role": "zero_shot_input_resolution_not_physical_convergence",
            "threshold_diagnostic_status": "pending_frozen_winner_only",
            "next_gate": "final_report_and_human_understanding_record",
        },
    )
    _write_json(
        campaign / "run.meta.json",
        {
            "script": "python -m phc2d.forward.full_cell_upgrade_50k_evaluation finalize",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_hash": config["plan_hash"],
        },
    )
    _write_csv(campaign / "history.csv", completed)
    (campaign / "run.log").write_text("formal frozen upgrade 50k evaluation finalized.\n", encoding="utf-8")
    return campaign


def plan_threshold_diagnostic(
    campaign_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Freeze five threshold proxies for the native graded winner only."""

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    parent_result = _read_json(campaign / "results.json")
    if parent_result.get("status") != "formal_upgrade_50k_evaluation_success":
        raise RuntimeError("threshold planning requires a finalized frozen 50k evaluation")
    native = _read_csv(campaign / "native_test_summary.csv")
    candidates = [
        row for row in native
        if row["source_group"] == "upgrade50k" and row["target_lane"] == "G"
    ]
    if len(candidates) != 5:
        raise RuntimeError(f"threshold winner requires five graded upgrade candidates, got {len(candidates)}")
    winner = min(candidates, key=lambda row: (float(row["mean_mre_pct"]), row["model"]))
    inventory = _read_csv(campaign / "checkpoint_inventory.csv")
    sources = [
        row for row in inventory
        if row["source_group"] == "upgrade50k"
        and row["model"] == winner["model"]
        and row["source_lane"] == winner["source_lane"]
    ]
    sources.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in sources] != [0, 1, 2]:
        raise RuntimeError("threshold winner lacks exactly three frozen seed checkpoints")
    threshold_dir = campaign / "threshold_diagnostic"
    if threshold_dir.exists() and any(threshold_dir.iterdir()):
        raise FileExistsError(f"threshold campaign is not empty: {threshold_dir}")
    threshold_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source in sources:
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
            label = f"t{int(round(threshold * 10)):02d}"
            rows.append(
                {
                    **source,
                    "threshold": threshold,
                    "threshold_label": label,
                    "target_lane": "G",
                    "evaluation_resolution": 64,
                    "contract": "threshold_representation_proxy",
                    "planned_status": "planned",
                    "shard_id": int(source["seed"]),
                    "evaluation_id": f"threshold_{winner['model']}_s{source['seed']}_{label}",
                }
            )
    _write_csv(threshold_dir / "threshold_plan.csv", rows)
    parent_config = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    config = {
        "schema_version": "phc2d.full_cell_upgrade_50k_threshold.v1",
        "parent_campaign": _relative(root, campaign),
        "parent_plan_hash": parent_config["plan_hash"],
        "winner_model": winner["model"],
        "winner_head": winner["head"],
        "winner_source_lane": winner["source_lane"],
        "winner_native_graded_mean_mre_pct": float(winner["mean_mre_pct"]),
        "winner_selection_rule": "lowest_frozen_native_graded_test_mean_among_five_upgrade_models",
        "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "datasets": {"G": parent_config["datasets"]["G"]},
        "plan_hash": hashlib.sha256((threshold_dir / "threshold_plan.csv").read_bytes()).hexdigest(),
        "evidence_boundary": "representation_proxy_not_physical_equivalence_or_native_promotion",
    }
    (threshold_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(
        threshold_dir / "campaign_status.json",
        {
            "status": "threshold_plan_ready",
            "winner_model": winner["model"],
            "planned_cells": 15,
            "test_influence": "winner_selection_only_after_native_promotion_frozen",
            "next_gate": "authorized_threshold_diagnostic_submission",
        },
    )
    return threshold_dir


def run_threshold_shard(
    threshold_dir: str | Path,
    *,
    shard_id: int,
    repository_root: str | Path = ".",
) -> Path:
    """Evaluate one winner seed across the five frozen threshold proxies."""

    root = Path(repository_root).resolve()
    threshold = _resolve(root, threshold_dir)
    config = yaml.safe_load((threshold / "config.resolved.yaml").read_text(encoding="utf-8"))
    rows = [row for row in _read_csv(threshold / "threshold_plan.csv") if int(row["shard_id"]) == int(shard_id)]
    if len(rows) != 5 or len({row["source_id"] for row in rows}) != 1:
        raise RuntimeError(f"threshold shard {shard_id} must contain five cells for one source")
    out = threshold / "shards" / f"shard_{int(shard_id):02d}"
    out.mkdir(parents=True, exist_ok=False)
    import torch

    device_info = resolve_device(
        DeviceConfig(
            device="cuda:0",
            require_accelerator=True,
            precision="fp32",
            amp_enabled=False,
            tf32_enabled=False,
            num_workers=4,
            pin_memory=True,
        ),
        torch_module=torch,
    )
    torch.cuda.reset_peak_memory_stats(device_info.torch_device)
    checkpoint = _resolve(root, rows[0]["checkpoint_path"])
    if _sha256(checkpoint) != rows[0]["checkpoint_sha256"]:
        raise RuntimeError("threshold winner checkpoint hash changed")
    payload = _checkpoint_payload(checkpoint, device=device_info.torch_device, torch=torch)
    spec = get_model_spec(rows[0]["model"])
    model_kwargs = dict(payload["model_params"])
    if spec.accepts_k_points:
        model_kwargs["k_points"] = payload.get("k_points")
    model = build_model(
        spec.name,
        input_shape=tuple(payload["input_shape"]),
        output_shape=tuple(payload["output_shape"]),
        **model_kwargs,
    ).to(device_info.torch_device)
    model.load_state_dict(payload["state_dict"])
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    completed: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    evaluation_ids: list[str] = []
    dataset_indices: list[np.ndarray] = []
    sample_mre: list[np.ndarray] = []
    started = time.perf_counter()
    for row in sorted(rows, key=lambda item: float(item["threshold"])):
        dataset_contract = dict(config["datasets"]["G"])
        dataset_contract["sampling"] = "deterministic_threshold"
        dataset_contract["threshold"] = float(row["threshold"])
        metrics, per_band, indices, values = _evaluate_cell(
            model=model,
            model_spec=spec,
            payload=payload,
            row=row,
            dataset_contract=dataset_contract,
            root=root,
            device_info=device_info,
            torch=torch,
        )
        completed.append(
            {
                **row,
                **metrics,
                "status": "completed",
                "fine_tuned": False,
                "parameter_count": parameter_count,
            }
        )
        band_rows.extend(per_band)
        evaluation_ids.append(row["evaluation_id"])
        dataset_indices.append(indices)
        sample_mre.append(values)
        _write_csv(out / "history.csv", completed)
    _write_csv(out / "evaluation_cells.csv", completed)
    _write_csv(out / "per_band_metrics.csv", band_rows)
    with (out / "sample_metrics.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            evaluation_ids=np.asarray(evaluation_ids),
            dataset_indices=np.stack(dataset_indices),
            sample_mre_pct=np.stack(sample_mre),
        )
    wall = time.perf_counter() - started
    _write_json(
        out / "metrics.json",
        {
            "completed_cells": 5,
            "wall_time_total_s": wall,
            "eval_time_total_s": sum(float(row["eval_time_s"]) for row in completed),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_info.torch_device) / (1024**2)),
        },
    )
    _write_json(
        out / "results.json",
        {
            "status": "formal_threshold_diagnostic_shard_success",
            "shard_id": int(shard_id),
            "completed_cells": 5,
            "plan_hash": config["plan_hash"],
        },
    )
    _write_json(
        out / "run.meta.json",
        {
            "script": "python -m phc2d.forward.full_cell_upgrade_50k_evaluation run-threshold-shard",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            **device_info.metadata(),
        },
    )
    (out / "config.resolved.yaml").write_text(
        yaml.safe_dump({"threshold_campaign": _relative(root, threshold), "shard_id": int(shard_id)}),
        encoding="utf-8",
    )
    (out / "run.log").write_text("frozen winner threshold diagnostic shard completed.\n", encoding="utf-8")
    return out


def finalize_threshold_diagnostic(
    threshold_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Aggregate the three winner-seed threshold shards and close the proxy gate."""

    root = Path(repository_root).resolve()
    threshold = _resolve(root, threshold_dir)
    config = yaml.safe_load((threshold / "config.resolved.yaml").read_text(encoding="utf-8"))
    plan = _read_csv(threshold / "threshold_plan.csv")
    completed: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for shard_id in range(3):
        shard = threshold / "shards" / f"shard_{shard_id:02d}"
        required = (
            "run.log",
            "run.meta.json",
            "config.resolved.yaml",
            "results.json",
            "metrics.json",
            "history.csv",
            "evaluation_cells.csv",
            "per_band_metrics.csv",
            "sample_metrics.npz",
            "n40_profile.csv",
            "n40_profile.json",
        )
        missing = [name for name in required if not (shard / name).is_file() or not (shard / name).stat().st_size]
        if missing:
            raise RuntimeError(f"threshold shard {shard_id} is incomplete: {missing}")
        result = _read_json(shard / "results.json")
        metrics = _read_json(shard / "metrics.json")
        profile = _read_json(shard / "n40_profile.json")
        if result.get("status") != "formal_threshold_diagnostic_shard_success":
            raise RuntimeError(f"threshold shard {shard_id} failed")
        if result.get("plan_hash") != config["plan_hash"] or profile.get("status") != "profile_collected":
            raise RuntimeError(f"threshold shard {shard_id} identity/profile mismatch")
        rows = _read_csv(shard / "evaluation_cells.csv")
        for row in rows:
            row["evaluation_peak_gpu_memory_mb"] = float(metrics["peak_gpu_memory_mb"])
        completed.extend(rows)
        profiles.append({"shard_id": shard_id, **metrics, **profile})
    if len(completed) != 15 or {row["evaluation_id"] for row in completed} != {row["evaluation_id"] for row in plan}:
        raise RuntimeError("threshold shards do not exactly cover the frozen 15-cell plan")
    summary = _summary(
        completed,
        ("model", "head", "source_lane", "target_lane", "threshold", "threshold_label"),
    )
    _write_csv(threshold / "evaluation_cells.csv", completed)
    _write_csv(threshold / "threshold_summary.csv", summary)
    _write_csv(threshold / "resource_summary.csv", profiles)
    import matplotlib.pyplot as plt

    figures = threshold / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ordered = sorted(summary, key=lambda row: float(row["threshold"]))
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.errorbar(
        [float(row["threshold"]) for row in ordered],
        [float(row["mean_mre_pct"]) for row in ordered],
        yerr=[float(row["std_mre_pct"]) for row in ordered],
        marker="o",
        capsize=3,
    )
    axis.set_xlabel("Deterministic threshold")
    axis.set_ylabel("Bands 2–10 MRE (%)")
    axis.set_title("Frozen graded winner threshold proxy")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "threshold_proxy.png", dpi=180)
    figure.savefig(figures / "threshold_proxy.pdf")
    plt.close(figure)
    _write_json(
        threshold / "metrics.json",
        {
            "evaluation_cells": 15,
            "thresholds": 5,
            "seeds": 3,
            "sum_shard_wall_time_s": sum(float(row["wall_time_total_s"]) for row in profiles),
        },
    )
    _write_json(
        threshold / "results.json",
        {
            "status": "formal_threshold_representation_proxy_success",
            "winner_model": config["winner_model"],
            "evidence_role": "representation_proxy_not_physical_equivalence",
            "native_promotion_influence": False,
            "next_gate": "final_report_and_human_understanding_record",
        },
    )
    _write_json(
        threshold / "run.meta.json",
        {
            "script": "python -m phc2d.forward.full_cell_upgrade_50k_evaluation finalize-threshold",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_hash": config["plan_hash"],
        },
    )
    _write_csv(threshold / "history.csv", completed)
    (threshold / "run.log").write_text("formal threshold representation proxy finalized.\n", encoding="utf-8")
    parent = _resolve(root, config["parent_campaign"])
    parent_result = _read_json(parent / "results.json")
    parent_result["threshold_diagnostic_status"] = "formal_threshold_representation_proxy_success"
    parent_result["threshold_winner_model"] = config["winner_model"]
    parent_result["next_gate"] = "final_report_and_human_understanding_record"
    _write_json(parent / "results.json", parent_result)
    render_final_markdown_report(parent, threshold_dir=threshold, repository_root=root)
    return threshold


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(clean(value) for value in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _percent(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}%"


def _relative_percent(value: Any, digits: int = 2) -> str:
    return f"{100.0 * float(value):+.{digits}f}%"


def render_final_markdown_report(
    campaign_dir: str | Path,
    *,
    threshold_dir: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Render the machine-audited experiment tables into a Chinese Draft report."""

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    threshold = _resolve(root, threshold_dir)
    results = _read_json(campaign / "results.json")
    native = [row for row in _read_csv(campaign / "native_test_summary.csv") if row["source_group"] == "upgrade50k"]
    comparisons = _read_csv(campaign / "paired_bootstrap_comparisons.csv")
    promotion = _read_csv(campaign / "promotion_decisions.csv")
    cross = _read_csv(campaign / "cross_lane_summary.csv")
    resolution = _read_csv(campaign / "resolution_transfer_summary.csv")
    per_band = _read_csv(campaign / "native_per_band_summary.csv")
    threshold_summary = _read_csv(threshold / "threshold_summary.csv")
    direct = {
        (row["model"], row["lane"]): row
        for row in comparisons
        if row["comparison"] == "direct_baseline"
    }
    fno = {
        (row["model"], row["lane"]): row
        for row in comparisons
        if row["comparison"] == "current_best_fno_e200"
    }
    decisions = {row["model"]: row["status"] for row in promotion}

    native_rows = []
    for row in sorted(native, key=lambda item: (item["model"], item["target_lane"])):
        key = (row["model"], row["target_lane"])
        direct_row = direct.get(key)
        fno_row = fno[key]
        native_rows.append(
            (
                row["model"],
                row["target_lane"],
                _percent(row["seed_0"]),
                _percent(row["seed_1"]),
                _percent(row["seed_2"]),
                f"{_percent(row['mean_mre_pct'])} ± {_percent(row['std_mre_pct'])}",
                _relative_percent(direct_row["relative_improvement"]) if direct_row else "joint model",
                (
                    f"[{float(direct_row['ci95_low']):+.4f}, {float(direct_row['ci95_high']):+.4f}]"
                    if direct_row else "see joint comparison"
                ),
                _relative_percent(fno_row["relative_improvement"]),
                decisions[row["model"]],
            )
        )

    cross_rows = [
        (
            row["model"],
            f"{row['source_lane']}→{row['target_lane']}",
            _percent(row["seed_0"]),
            _percent(row["seed_1"]),
            _percent(row["seed_2"]),
            f"{_percent(row['mean_mre_pct'])} ± {_percent(row['std_mre_pct'])}",
        )
        for row in sorted(cross, key=lambda item: (item["model"], item["source_lane"]))
    ]
    resolution_rows = [
        (
            row["model"],
            row["target_lane"],
            row["evaluation_resolution"],
            _percent(row["mean_mre_pct"]),
            _percent(row["std_mre_pct"]),
        )
        for row in sorted(
            resolution,
            key=lambda item: (item["model"], item["target_lane"], int(item["evaluation_resolution"])),
        )
    ]
    band_groups: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for row in per_band:
        band_groups[(row["model"], row["lane"])][int(row["band_one_based"])] = _percent(row["mean_mre_pct"])
    band_rows = [
        (model, lane, *(values[index] for index in range(1, 11)))
        for (model, lane), values in sorted(band_groups.items())
    ]
    resource_rows = [
        (
            row["model"],
            row["target_lane"],
            "/".join(str(int(row[f"actual_stop_epoch_{seed}"])) for seed in range(3)),
            "/".join(str(row[f"best_validation_epoch_{seed}"]) for seed in range(3)),
            "/".join(str(row[f"epoch_100_decision_{seed}"]) for seed in range(3)),
            f"{float(row['train_time_s_mean']) / 3600.0:.2f}",
            f"{float(row['training_peak_gpu_memory_mib_max']):.1f}",
            f"{int(row['parameter_count']):,}",
            f"{float(row['checkpoint_bytes_mean']) / (1024**2):.2f}",
            f"{float(row['evaluation_peak_gpu_memory_mb_max']):.1f}",
            f"{float(row['eval_time_s_mean']):.2f}",
            f"{float(row['inference_time_per_sample_ms_mean']):.4f}",
            f"{float(row['throughput_samples_per_s_mean']):.2f}",
        )
        for row in sorted(native, key=lambda item: (item["model"], item["target_lane"]))
    ]
    threshold_rows = [
        (
            row["model"],
            row["threshold"],
            _percent(row["seed_0"]),
            _percent(row["seed_1"]),
            _percent(row["seed_2"]),
            f"{_percent(row['mean_mre_pct'])} ± {_percent(row['std_mre_pct'])}",
        )
        for row in sorted(threshold_summary, key=lambda item: float(item["threshold"]))
    ]
    independent_promotion_rows = [
        (
            row["model"],
            row["status"],
            _relative_percent(row["binary_relative_improvement"]),
            _relative_percent(row["graded_relative_improvement"]),
            f"{float(row['binary_ci95_high']):+.4f}",
            f"{float(row['graded_ci95_high']):+.4f}",
            row["parameter_budget_met"],
        )
        for row in promotion
        if row["model"] != "dual_lane_band_operator_v1"
    ]
    joint = next(row for row in promotion if row["model"] == "dual_lane_band_operator_v1")
    joint_promotion_rows = [
        (
            joint["status"],
            joint["best_binary_independent"],
            joint["best_graded_independent"],
            _relative_percent(joint["binary_relative_degradation"]),
            _relative_percent(joint["graded_relative_degradation"]),
            _relative_percent(joint["binary_worst_seed_relative_regression"]),
            _relative_percent(joint["graded_worst_seed_relative_regression"]),
            f"{100.0 * float(joint['parameter_fraction']):.2f}%",
            f"{100.0 * float(joint['checkpoint_storage_fraction']):.2f}%",
        )
    ]
    promoted = results.get("promoted_over_direct_baseline") or []
    overall = results.get("overall_better_than_current_fno_models") or []
    conclusion = (
        f"直接基线 promotion 通过模型：{', '.join(promoted) if promoted else '无'}；"
        f"同时在 binary/graded 两条 native lane 满足当前 FNO 总体胜出门的模型："
        f"{', '.join(overall) if overall else '无'}。"
    )
    content = f"""<!--
  Draft machine-rendered report for the Full-Wh upgrade 50k experiment.
  Author: fangtoast
  Email: fangtoast@foxmail.com
  Copyright (c) 2026 fangtoast. All rights reserved.
-->

<a id="full-cell-upgrade-50k-report"></a>

# Full-Cell 升级模型 50K 正式训练与验证报告

> **Status:** Draft machine-audited experimental report<br>
> **Primary layer:** L3 experiment/evidence<br>
> **Evidence campaign:** `{_relative(root, campaign)}`<br>
> **Threshold campaign:** `{_relative(root, threshold)}`

## 一、结论

{conclusion}

本结论只基于冻结 40k/5k/5k split 的 held-out surrogate test。cross-lane 是 domain-shift 诊断，32/128 是零样本输入分辨率诊断，threshold 是 representation proxy；三者均不替代 native 结果，也不构成 real-MPB 或 inverse replay 结论。

## 二、Native held-out test 主结果

{_markdown_table(('模型', 'Lane', 'Seed 0', 'Seed 1', 'Seed 2', 'Mean ± std', '相对直接基线', '直接基线 paired CI', '相对 FNO-e200', 'Promotion'), native_rows)}

## 三、Promotion 门逐项判定

### 3.1 独立升级模型相对直接基线

{_markdown_table(('模型', '判定', 'Binary relative improvement', 'Graded relative improvement', 'Binary paired CI 上界', 'Graded paired CI 上界', '参数预算门'), independent_promotion_rows)}

### 3.2 联合模型替代两个独立模型

{_markdown_table(('判定', 'Binary 最佳独立模型', 'Graded 最佳独立模型', 'Binary mean 退化', 'Graded mean 退化', 'Binary worst-seed 退化', 'Graded worst-seed 退化', '参数比例', 'Checkpoint 存储比例'), joint_promotion_rows)}

## 四、逐 Band 结果

{_markdown_table(('模型', 'Lane', 'Band 1', 'Band 2', 'Band 3', 'Band 4', 'Band 5', 'Band 6', 'Band 7', 'Band 8', 'Band 9', 'Band 10'), band_rows)}

## 五、Cross-lane 诊断

{_markdown_table(('模型', '方向', 'Seed 0', 'Seed 1', 'Seed 2', 'Mean ± std'), cross_rows)}

联合模型因同时在两条 lane 上训练，没有单源 cross-lane 方向，对应 6 项保留为 `not_applicable_joint_trained_both_lanes`。

## 六、32/128 零样本输入分辨率诊断

{_markdown_table(('模型', 'Lane', '输入分辨率', 'Mean MRE', 'Std'), resolution_rows)}

ResMLP 固定展平输入的 12 项记录为 `not_applicable_fixed_flatten_shape`，不按失败或零误差汇总。

## 七、冻结 graded 胜者 Threshold Proxy

{_markdown_table(('冻结胜者', 'Threshold', 'Seed 0', 'Seed 1', 'Seed 2', 'Mean ± std'), threshold_rows)}

该表在 native graded 胜者冻结后生成，不影响 promotion；threshold 结构不被视为与真实二值结构物理等价。

## 八、训练预算、参数与推理资源

{_markdown_table(('模型', 'Lane', '三 seed 停止 epoch', '三 seed best-validation epoch', '三 seed epoch-100 gate', '平均训练小时', '训练峰值显存 MiB', '参数量', 'Checkpoint MiB', '评估峰值显存 MiB', 'Eval time s', 'ms/sample', 'samples/s'), resource_rows)}

## 九、证据与复核入口

- `native_test_summary.csv`：native 三 seed 汇总。
- `paired_bootstrap_comparisons.csv`：10,000 次 paired sample bootstrap。
- `promotion_decisions.csv`：预注册 promotion 门。
- `native_per_band_summary.csv`：bands 1–10 结果。
- `cross_lane_summary.csv`：跨路线诊断。
- `resolution_transfer_summary.csv`：32/128 零样本输入诊断。
- `threshold_diagnostic/threshold_summary.csv`：冻结胜者 threshold proxy。
- `resource_summary.csv` 与各 shard `n40_profile.json`：资源证据。

## 十、人工理解记录

- 本报告解决的问题：以冻结 checkpoint 和 held-out test 汇总五个升级模型的主结果、对比、负结果及诊断边界。
- 关键规则：validation 决定 checkpoint；test 只用于冻结后的统计和 promotion；不从 20/100/200 基线预算中事后择优。
- 未解决的问题：本 campaign 不授予 `physically_verified`，也不自动授权 inverse surrogate 替换。
- 最近下一 gate：人工核对源制品、数值引用和 promotion 标签后，将 Draft 同步至项目结果报告。
"""
    report = campaign / "report.md"
    report.write_text(content, encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--training-campaign", required=True)
    plan.add_argument("--baseline-inventory", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--repository-root", default=".")
    plan.add_argument("--shard-count", type=int, default=8)
    plan.add_argument("--bootstrap-replicates", type=int, default=10_000)
    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--campaign-dir", required=True)
    shard.add_argument("--shard-id", type=int, required=True)
    shard.add_argument("--repository-root", default=".")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--campaign-dir", required=True)
    finalize.add_argument("--repository-root", default=".")
    threshold_plan = subparsers.add_parser("plan-threshold")
    threshold_plan.add_argument("--campaign-dir", required=True)
    threshold_plan.add_argument("--repository-root", default=".")
    threshold_shard = subparsers.add_parser("run-threshold-shard")
    threshold_shard.add_argument("--threshold-dir", required=True)
    threshold_shard.add_argument("--shard-id", type=int, required=True)
    threshold_shard.add_argument("--repository-root", default=".")
    threshold_finalize = subparsers.add_parser("finalize-threshold")
    threshold_finalize.add_argument("--threshold-dir", required=True)
    threshold_finalize.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(
            plan_evaluation_campaign(
                args.training_campaign,
                baseline_inventory=args.baseline_inventory,
                output_dir=args.output_dir,
                repository_root=args.repository_root,
                shard_count=args.shard_count,
                bootstrap_replicates=args.bootstrap_replicates,
            )
        )
    elif args.command == "run-shard":
        print(
            run_evaluation_shard(
                args.campaign_dir,
                shard_id=args.shard_id,
                repository_root=args.repository_root,
            )
        )
    elif args.command == "finalize":
        print(finalize_evaluation_campaign(args.campaign_dir, repository_root=args.repository_root))
    elif args.command == "plan-threshold":
        print(plan_threshold_diagnostic(args.campaign_dir, repository_root=args.repository_root))
    elif args.command == "run-threshold-shard":
        print(
            run_threshold_shard(
                args.threshold_dir,
                shard_id=args.shard_id,
                repository_root=args.repository_root,
            )
        )
    else:
        print(finalize_threshold_diagnostic(args.threshold_dir, repository_root=args.repository_root))


if __name__ == "__main__":
    main()


__all__ = [
    "build_evaluation_cells",
    "finalize_evaluation_campaign",
    "finalize_threshold_diagnostic",
    "plan_threshold_diagnostic",
    "plan_evaluation_campaign",
    "run_evaluation_shard",
    "run_threshold_shard",
]
