"""Freeze and execute the N40 K20/50K four-domain cross-domain campaign.

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
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

MODELS = (
    "mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2",
    "mlp_mixer_band_v1", "res_mlp_gated_v2", "convnext_band_v1", "tfno2d_band_v3",
)
ALL_MODELS = MODELS + ("dual_lane_band_operator_v1",)
DOMAINS = ("F50-B", "F50-G", "K20-B", "K20-G")
DOMAIN_LANE = {domain: domain[-1] for domain in DOMAINS}
DOMAIN_FAMILY = {domain: domain[:3] for domain in DOMAINS}
SHARDS = {"F50-B": 3, "F50-G": 3, "K20-B": 1, "K20-G": 1}
SHARD_OFFSETS = {"F50-B": 0, "F50-G": 3, "K20-B": 6, "K20-G": 7}
EXPECTED_CHECKPOINTS = 68
EXPECTED_CELLS = 272
BOOTSTRAP_REPLICATES = 10_000
JOINT_MODEL = "dual_lane_band_operator_v1"
FIXED_BATCHES = (64, 128, 256, 512)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = root / "DEPLOYED_GIT_COMMIT"
        return marker.read_text(encoding="utf-8").strip() if marker.is_file() else "unknown"


def _config(root: Path, config_path: str | Path) -> dict[str, Any]:
    path = _resolve(root, config_path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected YAML object: {path}")
    return dict(value)


def _audit_dataset(root: Path, domain: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    import h5py

    path = _resolve(root, str(contract["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = str(contract.get("sha256", ""))
    actual_hash = sha256_file(path)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(f"dataset hash mismatch for {domain}: {actual_hash} != {expected_hash}")
    grid_field = str(contract["grid_field"])
    with h5py.File(path, "r") as handle:
        required = {grid_field, "Wh", "k_points", "idx_train", "idx_val", "idx_test", "sample_id"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise RuntimeError(f"{domain} missing HDF5 fields: {missing}")
        test_idx = np.asarray(handle["idx_test"], dtype=np.int64)
        val_idx = np.asarray(handle["idx_val"], dtype=np.int64)
        grid_shape = tuple(int(v) for v in handle[grid_field].shape)
        wh_shape = tuple(int(v) for v in handle["Wh"].shape)
        if int(test_idx.size) != int(contract["test_size"]):
            raise RuntimeError(f"{domain} test size mismatch: {test_idx.size}")
        if len(grid_shape) != 3 or grid_shape[1:] != (64, 64) or wh_shape[1:] != (81, 10):
            raise RuntimeError(f"{domain} shape mismatch: grid={grid_shape}, Wh={wh_shape}")
        if not np.isfinite(handle[grid_field][:]).all() or not np.isfinite(handle["Wh"][:]).all():
            raise RuntimeError(f"{domain} has nonfinite source arrays")
        k_points = np.asarray(handle["k_points"], dtype=np.float32)
        if k_points.shape != (81, 2) or not np.isfinite(k_points).all():
            raise RuntimeError(f"{domain} invalid k_points")
        return {
            "domain": domain, "family": DOMAIN_FAMILY[domain], "lane": DOMAIN_LANE[domain],
            "path": _relative(root, path), "grid_field": grid_field,
            "dataset_sha256": actual_hash, "test_size": int(test_idx.size),
            "idx_val_hash": sha256_array(val_idx), "idx_test_hash": sha256_array(test_idx),
            "k_points_sha256": sha256_array(k_points), "input_test_sha256": sha256_array(handle[grid_field][test_idx]),
            "target_test_sha256": sha256_array(handle["Wh"][test_idx]),
            "shape": list(grid_shape), "target_shape": list(wh_shape),
        }


def _source_identity(row: Mapping[str, Any]) -> tuple[str, str, int]:
    lane = str(row.get("source_lane", row.get("train_lane", "")))
    return str(row["model"]), lane, int(row["seed"])


def _checkpoint_path(root: Path, row: Mapping[str, Any]) -> Path:
    path_value = row.get("checkpoint_path") or row.get("checkpoint")
    if not path_value:
        run_dir = row.get("run_dir")
        if not run_dir:
            raise RuntimeError(f"checkpoint path missing for {row}")
        path_value = str(Path(str(run_dir)) / "model.pt")
    return _resolve(root, str(path_value))


def _load_inventory(root: Path, path_value: str | Path, family: str, expected_seeds: set[int]) -> list[dict[str, Any]]:
    rows = _read_csv(_resolve(root, path_value))
    result: list[dict[str, Any]] = []
    for row in rows:
        model = str(row.get("model", ""))
        lane = str(row.get("source_lane", row.get("train_lane", "")))
        if model not in ALL_MODELS or lane not in {"B", "G", "J"}:
            raise RuntimeError(f"unsupported checkpoint identity: {row}")
        checkpoint = _checkpoint_path(root, row)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        expected_hash = str(row.get("checkpoint_sha256", ""))
        actual_hash = sha256_file(checkpoint)
        if not expected_hash or actual_hash != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
        result.append({
            **row, "source_family": family, "model": model, "source_lane": lane,
            "seed": int(row["seed"]), "checkpoint_path": _relative(root, checkpoint),
            "checkpoint_sha256": actual_hash, "checkpoint_role": row.get("checkpoint_role", "best_validation"),
        })
    identities = [_source_identity(row) for row in result]
    if len(result) != (17 if family == "K20" else 51) or len(set(identities)) != len(identities):
        raise RuntimeError(f"{family} inventory is not the exact frozen size/identity set")
    singles = {(model, lane, seed) for model, lane, seed in identities if lane in {"B", "G"}}
    expected_single = {(model, lane, seed) for model in MODELS for lane in ("B", "G") for seed in expected_seeds}
    if singles != expected_single:
        raise RuntimeError(f"{family} single-route inventory mismatch")
    joint = {(model, lane, seed) for model, lane, seed in identities if lane == "J"}
    expected_joint = {(JOINT_MODEL, "J", seed) for seed in expected_seeds}
    if joint != expected_joint:
        raise RuntimeError(f"{family} joint inventory mismatch")
    return result


def _make_cache(root: Path, campaign: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    import h5py

    source = _resolve(root, str(audit["path"]))
    cache = campaign / "test_cache" / f"{audit['domain']}.h5"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        if sha256_file(cache) == str(audit.get("cache_sha256", "")):
            return dict(audit)
    with h5py.File(source, "r") as src, h5py.File(cache, "w") as dst:
        indices = np.asarray(src["idx_test"], dtype=np.int64)
        dst.create_dataset("input", data=src[str(audit["grid_field"])][indices], dtype="f4")
        dst.create_dataset("Wh", data=src["Wh"][indices], dtype="f4")
        dst.create_dataset("test_indices", data=indices, dtype="i8")
        dst.create_dataset("k_points", data=np.asarray(src["k_points"], dtype="f4"))
        dst.attrs["source_dataset_sha256"] = str(audit["dataset_sha256"])
        dst.attrs["input_test_sha256"] = str(audit["input_test_sha256"])
        dst.attrs["target_test_sha256"] = str(audit["target_test_sha256"])
        dst.attrs["idx_test_hash"] = str(audit["idx_test_hash"])
    updated = dict(audit, cache_path=_relative(root, cache), cache_sha256=sha256_file(cache))
    return updated


def _cell_kind(source: Mapping[str, Any], target: str) -> str:
    if source["source_lane"] == "J":
        return "joint_source_family"
    same_family = source["source_family"] == DOMAIN_FAMILY[target]
    same_lane = source["source_lane"] == DOMAIN_LANE[target]
    return "native" if same_family and same_lane else (
        "cross-generator" if same_lane else ("cross-lane" if same_family else "generator+lane")
    )


def _weight(model: str, weights: Mapping[str, Any]) -> float:
    return float(weights.get(model, 1.0))


def _assign_shards(cells: list[dict[str, Any]], weights: Mapping[str, Any]) -> None:
    for domain in DOMAINS:
        domain_cells = [row for row in cells if row["target_domain"] == domain]
        loads = [0.0] * SHARDS[domain]
        ordered = sorted(domain_cells, key=lambda row: (-_weight(str(row["source_model"]), weights), str(row["checkpoint_id"])))
        for row in ordered:
            shard = min(range(SHARDS[domain]), key=lambda index: (loads[index], index))
            row["shard_id"] = SHARD_OFFSETS[domain] + shard
            row["work_weight"] = _weight(str(row["source_model"]), weights)
            loads[shard] += row["work_weight"]
        if sorted(int(row["shard_id"]) for row in domain_cells) != sorted(
            int(row["shard_id"]) for row in domain_cells
        ):
            raise AssertionError("unreachable deterministic shard assignment failure")


def build_evaluation_cells(checkpoints: Sequence[Mapping[str, Any]], weights: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if len(checkpoints) != EXPECTED_CHECKPOINTS:
        raise RuntimeError(f"expected {EXPECTED_CHECKPOINTS} checkpoints, got {len(checkpoints)}")
    cells: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_id = f"{checkpoint['source_family']}_{checkpoint['model']}_{checkpoint['source_lane']}_s{checkpoint['seed']}"
        for domain in DOMAINS:
            cells.append({
                "evaluation_id": f"crossdom_{len(cells):03d}", "checkpoint_id": checkpoint_id,
                "source_family": checkpoint["source_family"], "source_model": checkpoint["model"],
                "source_lane": checkpoint["source_lane"], "source_seed": checkpoint["seed"],
                "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "target_domain": domain, "target_family": DOMAIN_FAMILY[domain], "target_lane": DOMAIN_LANE[domain],
                "comparison_kind": _cell_kind(checkpoint, domain), "status": "planned",
            })
    _assign_shards(cells, weights or {})
    return cells


def _write_standard_artifacts(campaign: Path, config: Mapping[str, Any], status: str, metrics: Mapping[str, Any], results: Mapping[str, Any], *, repository_root: Path | None = None) -> None:
    _write_json(campaign / "run.meta.json", {"script": "python -m phc2d.forward.full_cell_cross_domain_n40", "deployed_git_commit": _commit(repository_root or Path.cwd()), "primary_layer": "L3", "scientific_boundary": "surrogate_held_out_cross_domain"})
    (campaign / "config.resolved.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_json(campaign / "metrics.json", dict(metrics))
    _write_json(campaign / "results.json", {"status": status, "scientific_boundary": "surrogate_held_out_cross_domain", **dict(results)})
    (campaign / "run.log").write_text(f"status={status}\nscientific_boundary=surrogate_held_out_cross_domain\n", encoding="utf-8")
    if not (campaign / "history.csv").is_file():
        (campaign / "history.csv").write_text("stage,status,timestamp\n", encoding="utf-8")
    (campaign / "figures").mkdir(exist_ok=True)


def freeze_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".", config_path: str | Path = "configs/research/full_cell_cross_domain_n40_v1.yaml") -> Path:
    root = Path(repository_root).resolve()
    campaign = _resolve(root, campaign_dir)
    if campaign.exists() and any(campaign.iterdir()):
        raise FileExistsError(f"campaign directory is not empty: {campaign}")
    campaign.mkdir(parents=True, exist_ok=False)
    config = _config(root, config_path)
    config_without_guard = {key: value for key, value in config.items() if key != "include_p4m"}
    if config.get("include_p4m") is not False or "p4m" in json.dumps(config_without_guard, ensure_ascii=False).lower():
        raise RuntimeError("p4m is forbidden in this campaign")
    domains = {domain: _audit_dataset(root, domain, contract) for domain, contract in config["datasets"].items()}
    if set(domains) != set(DOMAINS):
        raise RuntimeError("campaign must contain exactly F50-B/F50-G/K20-B/K20-G")
    k20 = _load_inventory(root, config["checkpoints"]["K20_inventory"], "K20", {0})
    f50 = _load_inventory(root, config["checkpoints"]["F50_inventory"], "F50", {0, 1, 2})
    checkpoints = k20 + f50
    if len(checkpoints) != EXPECTED_CHECKPOINTS:
        raise RuntimeError("checkpoint inventory is not exactly 68")
    checkpoint_rows = [{**row, "checkpoint_id": f"{row['source_family']}_{row['model']}_{row['source_lane']}_s{row['seed']}"} for row in checkpoints]
    _write_csv(campaign / "checkpoint_manifest.csv", checkpoint_rows)
    _write_csv(campaign / "dataset_manifest.csv", list(domains.values()))
    cached = {domain: _make_cache(root, campaign, audit) for domain, audit in domains.items()}
    _write_csv(campaign / "test_cache_manifest.csv", list(cached.values()))
    cells = build_evaluation_cells(checkpoint_rows, config.get("throughput_weights", {}))
    _write_csv(campaign / "evaluation_cells.csv", cells)
    _write_csv(campaign / "evaluation_plan.csv", cells)
    shard_rows = []
    for domain in DOMAINS:
        for shard in range(SHARDS[domain]):
            selected = [row for row in cells if row["target_domain"] == domain and int(row["shard_id"]) == shard]
            shard_rows.append({"target_domain": domain, "shard_id": shard, "cell_count": len(selected), "evaluation_ids": ";".join(row["evaluation_id"] for row in selected), "work_weight": sum(float(row["work_weight"]) for row in selected)})
    _write_csv(campaign / "shard_plan.csv", shard_rows)
    plan_hash = sha256_file(campaign / "evaluation_plan.csv")
    config["plan_hash"] = plan_hash
    config["deployed_git_commit"] = _commit(root)
    config["status"] = "frozen_preflight_required"
    _write_standard_artifacts(campaign, config, "frozen_preflight_required", {"planned_checkpoints": 68, "planned_cells": 272}, {"status": "frozen_preflight_required", "planned_checkpoints": 68, "planned_cells": 272, "test_metrics_read": False}, repository_root=root)
    _write_json(campaign / "campaign_status.json", {"status": "frozen_preflight_required", "plan_hash": plan_hash, "planned_checkpoints": 68, "planned_cells": 272, "test_metrics_read": False, "next_gate": "validation_batch_preflight"})
    return campaign


def _load_checkpoint(path: Path, device: Any, torch: Any) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping) or "state_dict" not in payload or "model_name" not in payload:
        raise RuntimeError(f"checkpoint identity payload incomplete: {path}")
    return dict(payload)


def _normalizer(payload: Mapping[str, Any], target_lane: str) -> tuple[np.ndarray, np.ndarray]:
    def as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return np.asarray(value, dtype=np.float32)

    if payload.get("target_normalizers"):
        item = payload["target_normalizers"].get("binary" if target_lane == "B" else "graded")
        if item is None:
            raise RuntimeError("joint checkpoint has no target normalizer for target lane")
        return as_numpy(item["mean"]), as_numpy(item["std"])
    if "target_mean" not in payload or "target_std" not in payload:
        raise RuntimeError("checkpoint source normalizer missing")
    return as_numpy(payload["target_mean"]), as_numpy(payload["target_std"])


def _predict(payload: Mapping[str, Any], x: np.ndarray, k_points: np.ndarray, lane: str, batch: int, device: Any, torch: Any) -> np.ndarray:
    from phc2d.models import build_model, get_model_spec

    model = build_model(str(payload["model_name"]), input_shape=tuple(payload["input_shape"]), output_shape=tuple(payload["output_shape"]), **dict(payload["model_params"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    spec = get_model_spec(str(payload["model_name"]))
    mean, std = _normalizer(payload, lane)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for begin in range(0, len(x), batch):
            xb = torch.from_numpy(x[begin:begin + batch]).to(device)
            kwargs: dict[str, Any] = {}
            if spec.accepts_k_points:
                kwargs["k_points"] = torch.from_numpy(k_points).to(device)
            if spec.accepts_lane_id:
                kwargs["lane_id"] = torch.full((len(xb),), 0 if lane == "B" else 1, dtype=torch.long, device=device)
            pred = model(xb, **kwargs).detach().cpu().numpy()
            predictions.append((pred * std + mean).astype(np.float32))
    return np.concatenate(predictions, axis=0)


def _preflight_one(payload: Mapping[str, Any], x: np.ndarray, k_points: np.ndarray, lane: str, device: Any, torch: Any) -> dict[str, Any]:
    selected = None
    reference = None
    peak = 0
    for batch in FIXED_BATCHES:
        try:
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            prediction = _predict(payload, x[: min(len(x), 128)], k_points, lane, batch, device, torch)
            if reference is None:
                reference = prediction
            delta = float(np.max(np.abs(prediction - reference)))
            peak = max(peak, int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0)
            if delta > 1.0e-6 or not np.isfinite(prediction).all():
                raise RuntimeError(f"batch numerical mismatch/nonfinite at {batch}: {delta}")
            selected = batch
        except (RuntimeError, ValueError) as exc:
            if "out of memory" in str(exc).lower() or "nonfinite" in str(exc).lower() or "mismatch" in str(exc).lower():
                break
            raise
    if selected is None:
        raise RuntimeError(f"no admissible batch for {payload['model_name']}")
    if peak > int(20 * 1024**3):
        raise RuntimeError(f"batch preflight exceeded 20 GiB: {peak}")
    return {"model": payload["model_name"], "lane": lane, "selected_batch": selected, "peak_memory_bytes": peak, "status": "pass"}


def run_preflight(campaign_dir: str | Path, *, repository_root: str | Path = ".", device: str = "cuda:0") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir)
    import h5py
    import torch

    device_obj = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    rows = _read_csv(campaign / "checkpoint_manifest.csv")
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["source_lane"]))
        if key in results:
            continue
        domain = "K20-B" if row["source_lane"] == "J" else next(domain for domain in DOMAINS if DOMAIN_FAMILY[domain] == row["source_family"] and DOMAIN_LANE[domain] == row["source_lane"])
        audit = next(item for item in _read_csv(campaign / "test_cache_manifest.csv") if item["domain"] == domain)
        with h5py.File(_resolve(root, audit["path"]), "r") as handle:
            val = np.asarray(handle[audit["grid_field"]][handle["idx_val"][:128]], dtype=np.float32)
            k_points = np.asarray(handle["k_points"], dtype=np.float32)
        payload = _load_checkpoint(_resolve(root, row["checkpoint_path"]), device_obj, torch)
        results[key] = _preflight_one(payload, val, k_points, str(row["source_lane"]), device_obj, torch)
    _write_csv(campaign / "batch_preflight_manifest.csv", list(results.values()))
    if not results or any(row["status"] != "pass" for row in results.values()):
        raise RuntimeError("batch preflight failed")
    status = _read_json(campaign / "campaign_status.json")
    status.update({"status": "evaluation_plan_ready", "batch_preflight": "pass", "next_gate": "dry_run_or_submit"})
    _write_json(campaign / "campaign_status.json", status)
    return campaign


def _metric_rows(prediction: np.ndarray, target: np.ndarray) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    if not np.isfinite(prediction).all():
        raise RuntimeError("nonfinite prediction")
    diff = prediction - target
    rel = np.abs(diff) / np.maximum(np.abs(target), 1.0e-8)
    sample_mre = np.mean(rel[:, :, 1:], axis=(1, 2)) * 100.0
    denom = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    defined = denom > 1.0e-12
    r2 = np.full_like(denom, np.nan, dtype=np.float64)
    r2[defined] = 1.0 - np.sum(diff * diff, axis=0)[defined] / denom[defined]
    weights = denom.copy(); weights[~defined] = 0.0
    r2_weighted = float(np.sum(np.nan_to_num(r2) * weights) / max(float(weights.sum()), 1.0e-12))
    per_band = []
    for band in range(target.shape[2]):
        for k_index in range(target.shape[1]):
            per_band.append({"band_one_based": band + 1, "k_index": k_index, "mre_pct": float(rel[:, k_index, band].mean() * 100.0), "r2": None if not defined[k_index, band] else float(r2[k_index, band]), "r2_status": "undefined_zero_target_variance" if not defined[k_index, band] else "defined"})
    metrics = {"bands_2_10_mre_pct": float(sample_mre.mean()), "full_matrix_mre_pct": float(rel.mean() * 100.0), "mae": float(np.abs(diff).mean()), "rmse": float(np.sqrt(np.mean(diff * diff))), "bands_2_10_r2_variance_weighted": r2_weighted, "band_order_violation_rate": float(np.mean(np.any(np.diff(prediction, axis=2) < 0, axis=2))), "sample_count": int(target.shape[0]), "undefined_r2_coordinates": int((~defined).sum())}
    return metrics, sample_mre, per_band


def run_shard(campaign_dir: str | Path, shard_id: int, *, repository_root: str | Path = ".", device: str = "cuda:0") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir); import h5py; import torch
    status = _read_json(campaign / "campaign_status.json")
    if status.get("status") not in {"evaluation_plan_ready", "evaluation_submitted"}:
        raise RuntimeError(f"campaign is not runnable: {status.get('status')}")
    plan = _read_csv(campaign / "evaluation_plan.csv")
    selected = [row for row in plan if int(row["shard_id"]) == int(shard_id)]
    if not selected:
        raise RuntimeError(f"empty or invalid shard: {shard_id}")
    cache_rows = {row["domain"]: row for row in _read_csv(campaign / "test_cache_manifest.csv")}
    batch_rows = {(row["model"], row["lane"]): row for row in _read_csv(campaign / "batch_preflight_manifest.csv")}
    device_obj = torch.device(device)
    out = campaign / "shards" / f"shard_{int(shard_id):02d}"; out.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    for cell in selected:
        cache = cache_rows[cell["target_domain"]]
        with h5py.File(_resolve(root, cache["cache_path"]), "r") as handle:
            x = np.asarray(handle["input"], dtype=np.float32); target = np.asarray(handle["Wh"], dtype=np.float32); k_points = np.asarray(handle["k_points"], dtype=np.float32)
        checkpoint = _resolve(root, cell["checkpoint_path"])
        if sha256_file(checkpoint) != cell["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint identity/hash drift: {checkpoint}")
        payload = _load_checkpoint(checkpoint, device_obj, torch)
        if str(payload["model_name"]) != cell["source_model"]:
            raise RuntimeError(f"checkpoint model identity mismatch: {checkpoint}")
        batch = int(batch_rows.get((cell["source_model"], cell["source_lane"]), {"selected_batch": 64})["selected_batch"])
        started = time.perf_counter(); prediction = _predict(payload, x, k_points, cell["target_lane"], batch, device_obj, torch); elapsed = time.perf_counter() - started
        metrics, sample_mre, per_band = _metric_rows(prediction, target)
        np.savez_compressed(out / f"{cell['evaluation_id']}.npz", sample_mre=sample_mre, prediction=prediction, target=target)
        _write_csv(out / f"{cell['evaluation_id']}_per_band.csv", [{"evaluation_id": cell["evaluation_id"], **item} for item in per_band])
        metric_rows.append({**cell, **metrics, "inference_time_s": elapsed, "inference_time_per_sample_ms": elapsed * 1000.0 / max(len(x), 1), "throughput_samples_per_s": len(x) / max(elapsed, 1.0e-12), "status": "completed", "source_normalizer_reused": True})
    _write_csv(out / "evaluation_cells.csv", metric_rows)
    _write_json(out / "results.json", {"status": "shard_success", "shard_id": int(shard_id), "cell_count": len(metric_rows), "nonfinite": False})
    return out


def _bootstrap(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed); difference = left - right; observed = float(difference.mean()); values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 250):
        indices = rng.integers(0, len(difference), size=(min(250, BOOTSTRAP_REPLICATES - start), len(difference)))
        values[start:start + len(indices)] = difference[indices].mean(axis=1)
    native_mean = float(right.mean()); cross_mean = float(left.mean())
    return {"mre_delta_pp": observed, "ci95_low_pp": float(np.quantile(values, 0.025)), "ci95_high_pp": float(np.quantile(values, 0.975)), "relative_degradation_pct": float(observed / max(abs(native_mean), 1.0e-12) * 100.0), "cross_native_ratio": float(cross_mean / max(abs(native_mean), 1.0e-12)), "bootstrap_replicates": BOOTSTRAP_REPLICATES}


def finalize_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir)
    plan = _read_csv(campaign / "evaluation_plan.csv"); expected = {row["evaluation_id"] for row in plan}; rows: list[dict[str, Any]] = []; arrays: dict[str, np.ndarray] = {}
    for shard_dir in sorted((campaign / "shards").glob("shard_*")):
        result_path = shard_dir / "evaluation_cells.csv"
        if not result_path.is_file():
            continue
        shard_rows = _read_csv(result_path); rows.extend(shard_rows)
        for row in shard_rows:
            archive = shard_dir / f"{row['evaluation_id']}.npz"
            if not archive.is_file():
                raise RuntimeError(f"missing cell archive: {archive}")
            arrays[row["evaluation_id"]] = np.load(archive)["sample_mre"]
    observed = [row["evaluation_id"] for row in rows]
    if len(rows) != EXPECTED_CELLS or set(observed) != expected or len(set(observed)) != EXPECTED_CELLS:
        raise RuntimeError(f"shard coverage is not exactly {EXPECTED_CELLS} unique cells")
    if any(row.get("status") != "completed" or not math.isfinite(float(row["bands_2_10_mre_pct"])) for row in rows):
        raise RuntimeError("nonfinite or incomplete cell result")
    _write_csv(campaign / "evaluation_cells.csv", rows); _write_csv(campaign / "per_band_metrics.csv", [{"evaluation_id": row["evaluation_id"], "bands_2_10_mre_pct": row["bands_2_10_mre_pct"]} for row in rows])
    _write_csv(campaign / "per_k_metrics.csv", [{"evaluation_id": row["evaluation_id"], "full_matrix_mre_pct": row["full_matrix_mre_pct"], "r2": row["bands_2_10_r2_variance_weighted"]} for row in rows])
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows: grouped[(row["target_domain"], row["comparison_kind"])].append(row)
    domain_summary = [{"target_domain": key[0], "comparison_kind": key[1], "cell_count": len(value), "bands_2_10_mre_pct_mean": float(np.mean([float(row["bands_2_10_mre_pct"]) for row in value]))} for key, value in sorted(grouped.items())]
    _write_csv(campaign / "domain_summary.csv", domain_summary)
    seed_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        seed_groups[(row["source_family"], row["source_seed"])].append(row)
    _write_csv(campaign / "seed_summary.csv", [{"source_family": key[0], "source_seed": key[1], "cell_count": len(value), "bands_2_10_mre_pct_mean": float(np.mean([float(item["bands_2_10_mre_pct"]) for item in value]))} for key, value in sorted(seed_groups.items())])
    native: dict[tuple[str, str, str, str], dict[str, str]] = {(r["target_domain"], r["source_model"], r["source_seed"], r["source_lane"]): r for r in rows if r["comparison_kind"] == "native"}
    comparisons = []
    for row in rows:
        if row["comparison_kind"] not in {"cross-generator", "cross-lane", "generator+lane"} or row["source_model"] == JOINT_MODEL: continue
        target_native = native.get((row["target_domain"], row["source_model"], row["source_seed"], row["target_lane"]))
        if target_native is None: continue
        comparisons.append({"evaluation_id": row["evaluation_id"], "native_evaluation_id": target_native["evaluation_id"], "target_domain": row["target_domain"], "source_model": row["source_model"], "source_seed": row["source_seed"], "comparison_kind": row["comparison_kind"], **_bootstrap(arrays[row["evaluation_id"]], arrays[target_native["evaluation_id"]], seed=int(hashlib.sha256(row["evaluation_id"].encode()).hexdigest()[:8], 16))})
    _write_csv(campaign / "paired_bootstrap_comparisons.csv", comparisons)
    resources = []
    for row in rows: resources.append({"evaluation_id": row["evaluation_id"], "target_domain": row["target_domain"], "inference_time_s": row["inference_time_s"], "inference_time_per_sample_ms": row["inference_time_per_sample_ms"], "throughput_samples_per_s": row["throughput_samples_per_s"], "gpu_profile": f"profiles/shard_{int(row['shard_id']):02d}/n40_profile.json"})
    _write_csv(campaign / "resource_summary.csv", resources)
    metrics = {"planned_checkpoints": 68, "planned_cells": 272, "completed_cells": len(rows), "failed_cells": 0, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "inference_time_total_s": float(sum(float(row["inference_time_s"]) for row in rows))}
    _write_standard_artifacts(campaign, {"plan_hash": sha256_file(campaign / "evaluation_plan.csv")}, "evaluation_complete_pending_report", metrics, {"completed_cells": len(rows), "joint_source_family_excluded_from_cross_lane_conclusions": True, "p4m_included": False}, repository_root=root)
    _write_json(campaign / "campaign_status.json", {"status": "evaluation_complete_pending_report", "plan_hash": sha256_file(campaign / "evaluation_plan.csv"), "planned_cells": 272, "completed_cells": len(rows), "failed_cells": 0, "next_gate": "draft_report_review"})
    return campaign


def status_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir); status = _read_json(campaign / "campaign_status.json")
    plan = _read_csv(campaign / "evaluation_plan.csv") if (campaign / "evaluation_plan.csv").is_file() else []
    completed = []
    for path in (campaign / "shards").glob("shard_*/evaluation_cells.csv"):
        completed.extend(_read_csv(path))
    summary = {**status, "planned_cells": len(plan), "completed_cells": len(completed), "missing_cells": len({row["evaluation_id"] for row in plan} - {row["evaluation_id"] for row in completed})}
    _write_json(campaign / "status_snapshot.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "preflight", "dry-run", "submit", "status", "run-shard", "finalize"):
        item = sub.add_parser(name); item.add_argument("--campaign-dir", type=Path, required=True); item.add_argument("--repository-root", type=Path, default=Path("."))
        if name == "freeze": item.add_argument("--config", type=Path, default=Path("configs/research/full_cell_cross_domain_n40_v1.yaml"))
        if name in {"preflight"}: item.add_argument("--device", default="cuda:0")
        if name == "run-shard":
            item.add_argument("--shard-id", type=int, required=True)
            item.add_argument("--device", default="cuda:0")
        if name == "status": item.set_defaults(func=lambda a: print(json.dumps(status_campaign(a.campaign_dir, repository_root=a.repository_root), ensure_ascii=False, indent=2)))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "freeze": print(freeze_campaign(args.campaign_dir, repository_root=args.repository_root, config_path=args.config))
    elif args.command == "preflight": print(run_preflight(args.campaign_dir, repository_root=args.repository_root, device=args.device))
    elif args.command == "status": args.func(args)
    elif args.command == "run-shard": print(run_shard(args.campaign_dir, args.shard_id, repository_root=args.repository_root, device=args.device))
    elif args.command == "finalize": print(finalize_campaign(args.campaign_dir, repository_root=args.repository_root))
    elif args.command == "dry-run": print(f"campaign_dir={_resolve(args.repository_root.resolve(), args.campaign_dir)} array=0-7%8 gpu_4090 gres=gpu:1 cpus=6 time=04:00:00")
    elif args.command == "submit": raise SystemExit("submit is performed by jobs/paracloud_submit_full_cell_cross_domain.sh")


if __name__ == "__main__": main()


__all__ = ["DOMAINS", "MODELS", "SHARDS", "build_evaluation_cells", "freeze_campaign", "run_preflight", "run_shard", "finalize_campaign", "status_campaign"]
