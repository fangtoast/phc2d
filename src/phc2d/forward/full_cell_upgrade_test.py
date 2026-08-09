"""Evaluate validation-frozen Full-Wh upgrades and their direct 10k baselines.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phc2d.forward.full_cell import (
    _FullCellH5Dataset,
    _forward_model,
    _read_contract,
)
from phc2d.forward.full_cell_dual_lane import _lane_contracts
from phc2d.models import build_model, get_model_spec
from phc2d.training import DeviceConfig, resolve_device


DIRECT_BASELINES = {
    "mlp_mixer_band_v1": "mlp",
    "res_mlp_gated_v2": "res_mlp_wide",
    "convnext_band_v1": "lifted_cnn",
    "tfno2d_band_v3": "fno2d_band_v2",
    "dual_lane_band_operator_v1": "fno2d_band_v2",
}
SHORT_NAMES = {
    "mlp_mixer_band_v1": "MM",
    "res_mlp_gated_v2": "RM",
    "convnext_band_v1": "CX",
    "tfno2d_band_v3": "TF",
    "dual_lane_band_operator_v1": "DL",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _as_numpy(value: Any, *, dtype: Any = np.float32) -> np.ndarray:
    """Convert checkpoint tensors or arrays without depending on their device."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int = 2_000,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a non-empty finite vector")
    draws = rng.integers(0, values.size, size=(replicates, values.size))
    estimates = np.mean(values[draws], axis=1)
    return (
        float(np.mean(values)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def build_test_manifest(
    head_lock: str | Path,
    *,
    baseline_binary_manifest: str | Path,
    baseline_graded_manifest: str | Path,
    output_dir: str | Path,
    repository_root: str | Path = ".",
) -> Path:
    """Freeze 27 upgrade and 24 direct-baseline checkpoint test tasks."""

    root = Path(repository_root).resolve()
    lock_path = _resolve(root, head_lock)
    lock = _read_json(lock_path)
    if lock.get("status") != "head_selection_frozen" or lock.get("test_read_during_selection") is not False:
        raise RuntimeError("test manifest requires a validation-only frozen head lock")
    out = _resolve(root, output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for selection in lock["selections"]:
        model = str(selection["model"])
        lane = str(selection["lane"])
        head = str(selection["selected_head"])
        for selected_run in selection["selected_runs"]:
            seed = int(selected_run["seed"])
            source_run = _resolve(root, selected_run["run_dir"])
            task_id = f"UP_{SHORT_NAMES[model]}_{'Q' if head == 'k_conditioned' else 'D'}_{lane}_s{seed}"
            rows.append(
                {
                    "task_id": task_id,
                    "source_kind": "upgrade",
                    "model": model,
                    "head": head,
                    "lane": lane,
                    "seed": seed,
                    "source_run_dir": str(source_run.relative_to(root)),
                    "checkpoint_path": str((source_run / "model.pt").relative_to(root)),
                    "checkpoint_sha256": str(selected_run["checkpoint_sha256"]),
                }
            )
    if len(rows) != 27:
        raise RuntimeError(f"head lock must resolve 27 upgrade checkpoints, got {len(rows)}")
    for lane, manifest_value in (("B", baseline_binary_manifest), ("G", baseline_graded_manifest)):
        baseline_rows = _read_csv(_resolve(root, manifest_value))
        if len(baseline_rows) != 12:
            raise RuntimeError(f"{lane} baseline manifest must contain 12 rows, got {len(baseline_rows)}")
        identities: set[tuple[str, int]] = set()
        for baseline in baseline_rows:
            model = str(baseline["model"])
            seed = int(baseline["seed"])
            if model not in {"mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2"}:
                raise ValueError(f"unexpected {lane} baseline model {model!r}")
            if str(baseline.get("status")) != "formal_training_success":
                raise RuntimeError(f"baseline checkpoint is not a successful formal run: {model}/{lane}/s{seed}")
            if seed not in {0, 1, 2} or (model, seed) in identities:
                raise ValueError(f"invalid or duplicate {lane} baseline identity {model}/s{seed}")
            identities.add((model, seed))
            if str(baseline.get("test_executed", "false")).strip().lower() not in {"", "false", "0"}:
                raise RuntimeError(f"baseline selection manifest read test for {model}/{lane}/s{seed}")
            rows.append(
                {
                    "task_id": f"BL_{model}_{lane}_s{seed}",
                    "source_kind": "baseline",
                    "model": model,
                    "head": "legacy",
                    "lane": lane,
                    "seed": seed,
                    "source_run_dir": str(baseline["run_dir"]),
                    "checkpoint_path": str(baseline["checkpoint_path"]),
                    "checkpoint_sha256": str(baseline["checkpoint_sha256"]),
                }
            )
    if len(rows) != 51:
        raise RuntimeError(f"test manifest requires 51 checkpoint tasks, got {len(rows)}")
    for index, row in enumerate(rows):
        row["task_index"] = index
        row["result_dir"] = str((out / str(row["task_id"])).relative_to(root))
        checkpoint = _resolve(root, row["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if _sha256(checkpoint) != str(row["checkpoint_sha256"]):
            raise ValueError(f"checkpoint hash mismatch while freezing {row['task_id']}")
    ordered = [
        {key: row[key] for key in ("task_index", "task_id", "source_kind", "model", "head", "lane", "seed", "source_run_dir", "checkpoint_path", "checkpoint_sha256", "result_dir")}
        for row in rows
    ]
    _write_csv(out / "test_manifest.csv", ordered)
    (out / "selection_input.json").write_text(
        json.dumps(
            {
                "head_selection_lock": str(lock_path.relative_to(root)),
                "head_selection_lock_sha256": _sha256(lock_path),
                "selection_split": "validation",
                "evaluation_split": "test",
                "upgrade_checkpoint_count": 27,
                "baseline_checkpoint_count": 24,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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


def _evaluate_lane(
    model: Any,
    loader: Any,
    *,
    lane: str,
    lane_index: int | None,
    mean: Any,
    std: Any,
    model_spec: Any,
    k_points: Any,
    sample_indices: Sequence[int],
    device: Any,
    torch: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    squared = 0.0
    absolute = 0.0
    full_relative = 0.0
    count = 0
    per_band_sum: np.ndarray | None = None
    per_band_count = 0
    per_sample_rows: list[dict[str, Any]] = []
    sample_offset = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
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
            prediction = prediction_norm * std + mean
            target = yb * std + mean
            pred_np = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
            target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
            diff = pred_np - target_np
            relative = np.abs(diff) / np.maximum(np.abs(target_np), 1.0e-8)
            squared += float(np.sum(diff * diff))
            absolute += float(np.sum(np.abs(diff)))
            full_relative += float(np.sum(relative))
            count += int(diff.size)
            band_values = np.sum(relative, axis=(0, 1))
            per_band_sum = band_values if per_band_sum is None else per_band_sum + band_values
            per_band_count += int(diff.shape[0] * diff.shape[1])
            sample_mre = 100.0 * np.mean(relative[..., 1:], axis=(1, 2))
            sample_full_mre = 100.0 * np.mean(relative, axis=(1, 2))
            ordering = np.mean(np.diff(pred_np, axis=-1) < 0.0, axis=(1, 2))
            if sample_offset + int(diff.shape[0]) > len(sample_indices):
                raise RuntimeError("test loader produced more samples than the frozen index contract")
            for local_index in range(diff.shape[0]):
                per_sample_rows.append(
                    {
                        "lane": lane,
                        "sample_order": sample_offset + local_index,
                        "dataset_index": int(sample_indices[sample_offset + local_index]),
                        "bands_2_10_mre_pct": float(sample_mre[local_index]),
                        "full_matrix_mre_pct": float(sample_full_mre[local_index]),
                        "ordering_violation_rate": float(ordering[local_index]),
                    }
                )
            sample_offset += int(diff.shape[0])
    if sample_offset != len(sample_indices):
        raise RuntimeError(
            f"test loader produced {sample_offset} samples, expected {len(sample_indices)}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    assert per_band_sum is not None
    per_band_rows = [
        {"lane": lane, "band_one_based": index + 1, "mre_pct": float(100.0 * value / per_band_count)}
        for index, value in enumerate(per_band_sum)
    ]
    headline = [row["bands_2_10_mre_pct"] for row in per_sample_rows]
    metrics = {
        "lane": lane,
        "test_bands_2_10_mre_pct": float(statistics.fmean(headline)),
        "test_full_matrix_mre_pct": 100.0 * full_relative / max(count, 1),
        "test_mae": absolute / max(count, 1),
        "test_rmse": float(np.sqrt(squared / max(count, 1))),
        "test_ordering_violation_rate": float(
            statistics.fmean(float(row["ordering_violation_rate"]) for row in per_sample_rows)
        ),
        "n_test": len(per_sample_rows),
        "eval_time_s": elapsed,
        "inference_time_per_sample_ms": 1000.0 * elapsed / max(len(per_sample_rows), 1),
        "throughput_samples_per_s": len(per_sample_rows) / max(elapsed, 1.0e-12),
    }
    return metrics, per_band_rows, per_sample_rows


def run_test_task(
    manifest_path: str | Path,
    *,
    task_index: int,
    repository_root: str | Path = ".",
) -> Path:
    """Evaluate one frozen checkpoint on its native held-out test split."""

    root = Path(repository_root).resolve()
    manifest = _read_csv(_resolve(root, manifest_path))
    matches = [row for row in manifest if int(row["task_index"]) == int(task_index)]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve test task index {task_index}")
    task = matches[0]
    out = _resolve(root, task["result_dir"])
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = _resolve(root, task["checkpoint_path"])
    actual_hash = _sha256(checkpoint)
    if actual_hash != task["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch for {task['task_id']}")

    import torch
    from torch.utils.data import DataLoader

    initial = _checkpoint_payload(checkpoint, device="cpu", torch=torch)
    resolved = dict(initial["resolved_config"])
    runtime = dict(resolved["runtime"])
    device_info = resolve_device(DeviceConfig.from_mapping(runtime), torch_module=torch)
    device = device_info.torch_device
    if not device_info.is_cuda:
        raise RuntimeError("formal frozen-checkpoint test requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    payload = _checkpoint_payload(checkpoint, device=device, torch=torch)
    spec = get_model_spec(str(payload["model_name"]))
    model_kwargs = dict(payload["model_params"])
    if spec.accepts_k_points:
        model_kwargs["k_points"] = payload.get("k_points")
    model = build_model(
        spec.name,
        input_shape=tuple(payload["input_shape"]),
        output_shape=tuple(payload["output_shape"]),
        **model_kwargs,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    k_points = (
        torch.as_tensor(payload["k_points"], dtype=torch.float32, device=device)
        if spec.accepts_k_points and payload.get("k_points") is not None
        else None
    )
    lanes = ("B", "G") if str(task["lane"]) == "J" else (str(task["lane"]),)
    evaluations: list[dict[str, Any]] = []
    all_band_rows: list[dict[str, Any]] = []
    all_sample_rows: list[dict[str, Any]] = []
    contract_records: list[dict[str, Any]] = []
    for lane in lanes:
        if str(task["lane"]) == "J":
            contracts = _lane_contracts(resolved)
            lane_name = "binary" if lane == "B" else "graded"
            contract = contracts[lane_name]
            normalizer = payload["target_normalizers"][lane_name]
            mean_np = _as_numpy(normalizer["mean"])
            std_np = _as_numpy(normalizer["std"])
            lane_index = 0 if lane == "B" else 1
            representation = dict(resolved["representation"])
        else:
            contract = _read_contract(resolved)
            mean_np = _as_numpy(payload["target_mean"])
            std_np = _as_numpy(payload["target_std"])
            lane_index = None
            representation = dict(resolved["representation"])
        if not np.allclose(mean_np, contract["target_mean"], rtol=1.0e-5, atol=1.0e-6):
            raise ValueError(f"checkpoint target mean does not match frozen train split for {task['task_id']}/{lane}")
        if not np.allclose(std_np, contract["target_std"], rtol=1.0e-5, atol=1.0e-6):
            raise ValueError(f"checkpoint target std does not match frozen train split for {task['task_id']}/{lane}")
        if payload.get("k_points_sha256") and str(payload["k_points_sha256"]) != str(contract["k_points_sha256"]):
            raise ValueError(f"checkpoint k-point hash mismatch for {task['task_id']}/{lane}")
        contract_records.append(
            {
                "lane": lane,
                "dataset_path": str(contract["path"]),
                "dataset_sha256": str(contract["dataset_sha256"]),
                "idx_test_hash": str(contract["idx_test_hash"]),
                "test_count": int(len(contract["idx_test"])),
                "k_points_sha256": str(contract["k_points_sha256"]),
            }
        )
        dataset = _FullCellH5Dataset(
            path=contract["path"],
            indices=contract["idx_test"],
            grid_field=contract["grid_field"],
            layout=str(representation["layout"]),
            sampling=representation.get("sampling", "continuous"),
            threshold=representation.get("threshold"),
            target_mean=mean_np,
            target_std=std_np,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(resolved["training"].get("micro_batch_size", 64)),
            shuffle=False,
            num_workers=int(device_info.num_workers),
            pin_memory=bool(device_info.pin_memory),
        )
        metrics, band_rows, sample_rows = _evaluate_lane(
            model,
            loader,
            lane=lane,
            lane_index=lane_index,
            mean=torch.as_tensor(mean_np, dtype=torch.float32, device=device),
            std=torch.as_tensor(std_np, dtype=torch.float32, device=device),
            model_spec=spec,
            k_points=k_points,
            sample_indices=contract["idx_test"],
            device=device,
            torch=torch,
        )
        evaluations.append(metrics)
        all_band_rows.extend(band_rows)
        all_sample_rows.extend(sample_rows)
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    metrics_payload = {
        "evaluations": evaluations,
        "data_contracts": contract_records,
        "peak_gpu_memory_mb": peak_mb,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    _write_csv(out / "per_band_metrics.csv", all_band_rows)
    _write_csv(out / "per_sample_metrics.csv", all_sample_rows)
    (out / "history.csv").write_text("evaluation_only,no_training_history\ntrue,true\n", encoding="utf-8")
    (out / "config.resolved.yaml").write_text(
        json.dumps(
            {
                "evaluation": {
                    "split": "test",
                    "selection_frozen": True,
                    "task_id": task["task_id"],
                    "checkpoint_path": task["checkpoint_path"],
                    "checkpoint_sha256": actual_hash,
                },
                "training_config_from_checkpoint": resolved,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps(
            {
                "status": "frozen_checkpoint_test_success",
                "test_executed": True,
                "selection_influence": False,
                "selection_was_frozen_before_test": True,
                "evaluation_data_contracts": contract_records,
                "task_id": task["task_id"],
                "source_kind": task["source_kind"],
                "model": task["model"],
                "head": task["head"],
                "lane": task["lane"],
                "seed": int(task["seed"]),
                "source_run_dir": task["source_run_dir"],
                "checkpoint_sha256": actual_hash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "run.meta.json").write_text(
        json.dumps(
            {
                "script": "python -m phc2d.forward.full_cell_upgrade_test run-task",
                "task_index": int(task_index),
                "evaluation_split": "test",
                "device_name": device_info.device_name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "run.log").write_text("frozen checkpoint native test completed.\n", encoding="utf-8")
    return out


def finalize_test_campaign(
    campaign_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Aggregate test results and freeze the preregistered 50k admission lock."""

    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    manifest = _read_csv(campaign / "test_manifest.csv")
    if len(manifest) != 51:
        raise RuntimeError(f"expected 51 test tasks, got {len(manifest)}")
    runs: list[dict[str, Any]] = []
    sample_errors: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for task in manifest:
        result_dir = _resolve(root, task["result_dir"])
        result = _read_json(result_dir / "results.json")
        metrics = _read_json(result_dir / "metrics.json")
        if result.get("status") != "frozen_checkpoint_test_success":
            raise RuntimeError(f"test task did not succeed: {task['task_id']}")
        if str(result.get("checkpoint_sha256")) != str(task["checkpoint_sha256"]):
            raise RuntimeError(f"checkpoint identity changed for {task['task_id']}")
        for required in ("per_band_metrics.csv", "per_sample_metrics.csv", "history.csv", "config.resolved.yaml", "run.meta.json", "run.log"):
            if not (result_dir / required).is_file():
                raise FileNotFoundError(result_dir / required)
        for evaluation in metrics["evaluations"]:
            lane = str(evaluation["lane"])
            runs.append(
                {
                    "task_id": task["task_id"],
                    "source_kind": task["source_kind"],
                    "model": task["model"],
                    "head": task["head"],
                    "lane": lane,
                    "seed": int(task["seed"]),
                    **{key: value for key, value in evaluation.items() if key != "lane"},
                    "parameter_count": int(metrics["parameter_count"]),
                    "peak_gpu_memory_mb": float(metrics["peak_gpu_memory_mb"]),
                    "result_dir": task["result_dir"],
                }
            )
            sample_rows = [
                row for row in _read_csv(result_dir / "per_sample_metrics.csv")
                if str(row["lane"]) == lane
            ]
            sample_rows.sort(key=lambda row: int(row["sample_order"]))
            sample_errors[(str(task["source_kind"]), str(task["model"]), lane, int(task["seed"]))] = (
                np.asarray([int(row["dataset_index"]) for row in sample_rows], dtype=np.int64),
                np.asarray([float(row["bands_2_10_mre_pct"]) for row in sample_rows], dtype=np.float64),
            )
    if len(runs) != 54:
        raise RuntimeError(f"expected 54 native test evaluation rows, got {len(runs)}")
    _write_csv(campaign / "test_runs.csv", runs)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in runs:
        key = (str(row["source_kind"]), str(row["model"]), str(row["head"]), str(row["lane"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items()):
        members.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in members] != [0, 1, 2]:
            raise RuntimeError(f"test summary group {key} does not contain seeds 0,1,2")
        values = [float(row["test_bands_2_10_mre_pct"]) for row in members]
        summaries.append(
            {
                "source_kind": key[0],
                "model": key[1],
                "head": key[2],
                "lane": key[3],
                "seed_0": values[0],
                "seed_1": values[1],
                "seed_2": values[2],
                "test_mean": statistics.fmean(values),
                "test_std": statistics.stdev(values),
                "worst_seed": max(values),
                "parameter_count": int(members[0]["parameter_count"]),
                "peak_gpu_memory_mb_max": max(float(row["peak_gpu_memory_mb"]) for row in members),
            }
        )
    _write_csv(campaign / "test_summary.csv", summaries)
    lookup = {
        (str(row["source_kind"]), str(row["model"]), str(row["lane"])): row
        for row in summaries
    }
    seed_mean_errors: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for source_kind, model, _head, lane in grouped:
        arrays = [sample_errors[(source_kind, model, lane, seed)] for seed in (0, 1, 2)]
        reference_indices = arrays[0][0]
        if any(not np.array_equal(reference_indices, item[0]) for item in arrays[1:]):
            raise RuntimeError(f"test sample order changed across seeds for {source_kind}/{model}/{lane}")
        seed_mean_errors[(source_kind, model, lane)] = (
            reference_indices,
            np.mean(np.stack([item[1] for item in arrays], axis=0), axis=0),
        )
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_by_model_lane: dict[tuple[str, str], dict[str, float]] = {}
    seed_sequence = np.random.SeedSequence(20260809)
    comparison_keys = [(model, lane) for model in DIRECT_BASELINES for lane in ("B", "G")]
    for child, (model, lane) in zip(seed_sequence.spawn(len(comparison_keys)), comparison_keys):
        baseline_model = DIRECT_BASELINES[model]
        upgrade_indices, upgrade_values = seed_mean_errors[("upgrade", model, lane)]
        baseline_indices, baseline_values = seed_mean_errors[("baseline", baseline_model, lane)]
        if not np.array_equal(upgrade_indices, baseline_indices):
            raise RuntimeError(f"upgrade and baseline do not share the frozen test split for {model}/{lane}")
        estimate, ci_low, ci_high = _bootstrap_mean_ci(
            upgrade_values - baseline_values,
            rng=np.random.default_rng(child),
        )
        record = {
            "model": model,
            "lane": lane,
            "baseline_model": baseline_model,
            "upgrade_minus_baseline_mre_pct_points": estimate,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "sample_count": int(upgrade_values.size),
            "bootstrap_replicates": 2_000,
            "paired": True,
        }
        bootstrap_rows.append(record)
        bootstrap_by_model_lane[(model, lane)] = record
    _write_csv(campaign / "paired_bootstrap_ci.csv", bootstrap_rows)
    decisions: list[dict[str, Any]] = []
    for model in DIRECT_BASELINES:
        baseline_model = DIRECT_BASELINES[model]
        lane_decisions: list[dict[str, Any]] = []
        for lane in ("B", "G"):
            upgrade = lookup[("upgrade", model, lane)]
            baseline = lookup[("baseline", baseline_model, lane)]
            upgrade_mean = float(upgrade["test_mean"])
            baseline_mean = float(baseline["test_mean"])
            relative = (baseline_mean - upgrade_mean) / baseline_mean
            bootstrap = bootstrap_by_model_lane[(model, lane)]
            lane_decisions.append(
                {
                    "lane": lane,
                    "upgrade_test_mean": upgrade_mean,
                    "baseline_model": baseline_model,
                    "baseline_test_mean": baseline_mean,
                    "relative_improvement": relative,
                    "upgrade_minus_baseline_mre_pct_points": bootstrap["upgrade_minus_baseline_mre_pct_points"],
                    "paired_ci95_low": bootstrap["ci95_low"],
                    "paired_ci95_high": bootstrap["ci95_high"],
                }
            )
        improvements = [float(item["relative_improvement"]) for item in lane_decisions]
        parameter_count = int(lookup[("upgrade", model, "B")]["parameter_count"])
        capacity_ok = 2_000_000 <= parameter_count <= 8_000_000
        if model == "dual_lane_band_operator_v1":
            performance_ok = all(value >= -0.05 for value in improvements)
            admitted = performance_ok and capacity_ok
            reason = (
                "both_native_lanes_not_worse_than_fno_by_5pct"
                if admitted
                else "parameter_budget_not_met" if not capacity_ok else "joint_native_regression_over_5pct"
            )
        else:
            performance_ok = max(improvements) >= 0.05 and min(improvements) >= -0.05
            admitted = performance_ok and capacity_ok
            reason = (
                "one_lane_improves_5pct_other_not_worse_5pct"
                if admitted
                else "parameter_budget_not_met" if not capacity_ok else "direct_baseline_gate_not_met"
            )
        decisions.append(
            {
                "model": model,
                "direct_baseline": baseline_model,
                "status": "admitted" if admitted else "not_admitted",
                "reason": reason,
                "parameter_count": parameter_count,
                "parameter_budget_2m_8m_met": capacity_ok,
                "lanes": lane_decisions,
            }
        )
    lock = {
        "schema_version": "phc2d.full_cell_upgrade_50k_admission.v1",
        "status": "50k_admission_frozen",
        "selection_split": "validation",
        "evaluation_split": "test_after_head_freeze",
        "test_influenced_head_selection": False,
        "test_task_count": len(manifest),
        "test_evaluation_row_count": len(runs),
        "decisions": decisions,
    }
    (campaign / "50k_admission.lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    (campaign / "results.json").write_text(
        json.dumps(
            {
                "status": "10k_frozen_head_test_and_50k_admission_success",
                "admitted_models": [item["model"] for item in decisions if item["status"] == "admitted"],
                "not_admitted_models": [item["model"] for item in decisions if item["status"] != "admitted"],
                "next_gate": "50k_dataset_hash_preflight_and_separate_submit_authorization",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return campaign


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--head-lock", required=True)
    plan.add_argument("--baseline-binary-manifest", required=True)
    plan.add_argument("--baseline-graded-manifest", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--repository-root", default=".")
    task = subparsers.add_parser("run-task")
    task.add_argument("--manifest", required=True)
    task.add_argument("--task-index", type=int, required=True)
    task.add_argument("--repository-root", default=".")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--campaign-dir", required=True)
    finalize.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(
            build_test_manifest(
                args.head_lock,
                baseline_binary_manifest=args.baseline_binary_manifest,
                baseline_graded_manifest=args.baseline_graded_manifest,
                output_dir=args.output_dir,
                repository_root=args.repository_root,
            )
        )
    elif args.command == "run-task":
        print(run_test_task(args.manifest, task_index=args.task_index, repository_root=args.repository_root))
    else:
        print(finalize_test_campaign(args.campaign_dir, repository_root=args.repository_root))


if __name__ == "__main__":
    main()


__all__ = ["build_test_manifest", "finalize_test_campaign", "run_test_task"]
