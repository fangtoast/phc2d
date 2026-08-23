"""Frozen six-domain Full-Wh held-out surrogate evaluation for EXP-FWD-007.

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
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell_cross_domain_n40 import _metric_rows, _normalizer, _predict, sha256_array, sha256_file
from phc2d.models import build_model, describe_model_identity, get_model_spec

MODELS = ("mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2", "mlp_mixer_band_v1", "res_mlp_gated_v2", "convnext_band_v1", "tfno2d_band_v3")
JOINT_MODEL = "dual_lane_band_operator_v1"
DOMAINS = ("K5-B", "K5-G", "K20-B", "K20-G", "F50-B", "F50-G")
EXPECTED_CHECKPOINTS, EXPECTED_CELLS, EXPECTED_SHARDS, EXPECTED_PREFLIGHTS = 85, 510, 10, 54


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows: raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path); return value if value.is_absolute() else root / value


def _config(root: Path, path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(_resolve(root, path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping): raise TypeError("six-domain config must be a mapping")
    return dict(value)


def _commit(root: Path) -> str:
    marker = root / "DEPLOYED_GIT_COMMIT"
    return marker.read_text(encoding="utf-8").strip() if marker.is_file() else "unknown"


def _audit_dataset(root: Path, domain: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    import h5py
    path = _resolve(root, str(contract["path"])); actual = sha256_file(path)
    if actual != str(contract["sha256"]): raise RuntimeError(f"dataset hash drift: {domain}")
    grid = str(contract["grid_field"])
    with h5py.File(path, "r") as data:
        if tuple(data["Wh"].shape) != (10000, 81, 10) or tuple(data[grid].shape) != (10000, 64, 64): raise RuntimeError(f"shape drift: {domain}")
        test = np.asarray(data["idx_test"], dtype=np.int64); val = np.asarray(data["idx_val"], dtype=np.int64); k = np.asarray(data["k_points"], dtype=np.float32)
        if len(test) != int(contract["test_size"]) or k.shape != (81, 2): raise RuntimeError(f"split or k-grid drift: {domain}")
        return {"domain": domain, "family": domain.split("-")[0], "lane": domain[-1], "path": str(contract["path"]), "grid_field": grid,
                "dataset_sha256": actual, "idx_val_hash": sha256_array(val), "idx_test_hash": sha256_array(test), "k_points_sha256": sha256_array(k),
                "input_test_sha256": sha256_array(data[grid][test]), "target_test_sha256": sha256_array(data["Wh"][test]), "test_size": len(test)}


def _checkpoint_path(root: Path, row: Mapping[str, Any]) -> Path:
    value = row.get("checkpoint_path") or (str(Path(str(row["run_dir"])) / "model.pt") if row.get("run_dir") else "")
    if not value: raise RuntimeError(f"checkpoint path missing: {row}")
    return _resolve(root, str(value))


def _inventory(root: Path, path: str, family: str, seeds: set[int]) -> list[dict[str, Any]]:
    if path.startswith("PLACEHOLDER_"): raise RuntimeError(f"{family} inventory must be frozen before six-domain evaluation")
    rows: list[dict[str, Any]] = []
    for row in _read_csv(_resolve(root, path)):
        model, lane = str(row.get("model", "")), str(row.get("source_lane", row.get("train_lane", "")))
        checkpoint = _checkpoint_path(root, row); actual = sha256_file(checkpoint)
        if model not in MODELS + (JOINT_MODEL,) or lane not in {"B", "G", "J"}: raise RuntimeError(f"invalid model identity: {row}")
        if str(row.get("checkpoint_sha256", "")) != actual: raise RuntimeError(f"checkpoint hash drift: {checkpoint}")
        rows.append({**row, "source_family": family, "source_lane": lane, "source_seed": int(row["seed"]), "checkpoint_path": checkpoint.as_posix(), "checkpoint_sha256": actual})
    expected = 17 * len(seeds)
    identities = {(r["model"], r["source_lane"], r["source_seed"]) for r in rows}
    if len(rows) != expected or len(identities) != expected: raise RuntimeError(f"{family} inventory must contain {expected} unique checkpoints")
    required = {(model, lane, seed) for model in MODELS for lane in ("B", "G") for seed in seeds} | {(JOINT_MODEL, "J", seed) for seed in seeds}
    if identities != required: raise RuntimeError(f"{family} checkpoint matrix mismatch")
    return rows


def _cache(root: Path, campaign: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    import h5py
    cache = campaign / "test_cache" / f"{audit['domain']}.h5"
    with h5py.File(_resolve(root, audit["path"]), "r") as source, h5py.File(cache, "w") as target:
        indices = np.asarray(source["idx_test"], dtype=np.int64); target.create_dataset("input", data=source[audit["grid_field"]][indices], dtype="f4")
        target.create_dataset("Wh", data=source["Wh"][indices], dtype="f4"); target.create_dataset("k_points", data=np.asarray(source["k_points"], dtype="f4")); target.create_dataset("test_indices", data=indices, dtype="i8")
        target.attrs["source_dataset_sha256"] = audit["dataset_sha256"]
    return {**audit, "cache_path": cache.as_posix(), "cache_sha256": sha256_file(cache)}


def _kind(row: Mapping[str, Any], domain: str) -> str:
    if row["source_lane"] == "J": return "joint_source_family"
    family, lane = domain.split("-")
    return "native" if (row["source_family"], row["source_lane"]) == (family, lane) else ("same_family_cross_lane" if row["source_family"] == family else ("cross_generator_same_lane" if row["source_lane"] == lane else "generator_and_lane"))


def build_evaluation_cells(checkpoints: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(checkpoints) != EXPECTED_CHECKPOINTS: raise RuntimeError(f"expected {EXPECTED_CHECKPOINTS} checkpoints")
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_id = f"{checkpoint['source_family']}_{checkpoint['model']}_{checkpoint['source_lane']}_s{checkpoint['source_seed']}"
        for domain in DOMAINS:
            rows.append({"evaluation_id": f"sixdom_{len(rows):03d}", "checkpoint_id": checkpoint_id, "source_family": checkpoint["source_family"], "source_model": checkpoint["model"], "source_lane": checkpoint["source_lane"], "source_seed": checkpoint["source_seed"], "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"], "target_domain": domain, "target_family": domain.split("-")[0], "target_lane": domain[-1], "comparison_kind": _kind(checkpoint, domain), "status": "planned", "shard_id": len(rows) % EXPECTED_SHARDS})
    if len(rows) != EXPECTED_CELLS: raise AssertionError("six-domain count regression")
    return rows


def freeze_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".", config_path: str | Path = "configs/research/full_cell_k5_six_domain_n40_v1.yaml") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir); config = _config(root, config_path)
    if campaign.exists(): raise FileExistsError(campaign)
    campaign.mkdir(parents=True); audits = {domain: _audit_dataset(root, domain, contract) for domain, contract in config["datasets"].items()}
    if tuple(audits) != DOMAINS: raise RuntimeError("six domains must be K5/K20/F50 × B/G")
    if len({row["k_points_sha256"] for row in audits.values()}) != 1: raise RuntimeError("six-domain k-grid mismatch")
    checkpoints = _inventory(root, config["checkpoints"]["K5_inventory"], "K5", {0}) + _inventory(root, config["checkpoints"]["K20_inventory"], "K20", {0}) + _inventory(root, config["checkpoints"]["F50_inventory"], "F50", {0, 1, 2})
    cells = build_evaluation_cells(checkpoints); cached = [_cache(root, campaign, audit) for audit in audits.values()]
    _write_csv(campaign / "dataset_manifest.csv", list(audits.values())); _write_csv(campaign / "test_cache_manifest.csv", cached); _write_csv(campaign / "checkpoint_manifest.csv", checkpoints); _write_csv(campaign / "evaluation_plan.csv", cells)
    shards = [{"shard_id": shard, "cell_count": sum(int(row["shard_id"]) == shard for row in cells), "evaluation_ids": ";".join(row["evaluation_id"] for row in cells if int(row["shard_id"]) == shard)} for shard in range(EXPECTED_SHARDS)]
    _write_csv(campaign / "shard_plan.csv", shards); plan_hash = sha256_file(campaign / "evaluation_plan.csv")
    resolved = {**config, "plan_hash": plan_hash, "deployed_git_commit": _commit(root)}; (campaign / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_json(campaign / "campaign_status.json", {"status": "frozen_preflight_required", "planned_checkpoints": EXPECTED_CHECKPOINTS, "planned_cells": EXPECTED_CELLS, "planned_shards": EXPECTED_SHARDS, "plan_hash": plan_hash, "test_metrics_read": False, "next_gate": "54_identity_preflights"})
    return campaign


def _load_checkpoint(path: Path, device: Any, torch: Any) -> dict[str, Any]:
    try: value = torch.load(path, map_location=device, weights_only=True)
    except TypeError: value = torch.load(path, map_location=device)
    if not isinstance(value, Mapping) or "state_dict" not in value: raise RuntimeError(f"invalid checkpoint: {path}")
    return dict(value)


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    model = build_model(str(payload["model_name"]), input_shape=tuple(payload["input_shape"]), output_shape=tuple(payload["output_shape"]), **dict(payload["model_params"]))
    return dict(payload.get("model_identity") or describe_model_identity(model, str(payload["model_name"]), model_kwargs=dict(payload["model_params"])))


def run_preflight(campaign_dir: str | Path, *, repository_root: str | Path = ".", device: str = "cuda:0") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir); import h5py, torch
    device_obj = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu"); caches = {r["domain"]: r for r in _read_csv(campaign / "test_cache_manifest.csv")}; rows: list[dict[str, Any]] = []; seen: set[tuple[str, ...]] = set()
    for source in _read_csv(campaign / "checkpoint_manifest.csv"):
        lanes = ("B", "G") if source["source_lane"] == "J" else (source["source_lane"],)
        for lane in lanes:
            key = (source["source_family"], source["model"], source["source_lane"], lane)
            if key in seen: continue
            seen.add(key); domain = f"{source['source_family']}-{lane}"; cache = caches[domain]
            with h5py.File(cache["cache_path"], "r") as data: x, k = np.asarray(data["input"][:1], dtype=np.float32), np.asarray(data["k_points"], dtype=np.float32)
            payload = _load_checkpoint(Path(source["checkpoint_path"]), device_obj, torch); identity = _identity(payload)
            if identity.get("canonical_model_name") != source["model"] or not identity.get("factory_function_identity") or not identity.get("architecture_fingerprint"): raise RuntimeError("checkpoint model identity mismatch")
            prediction = _predict(payload, x, k, lane, 1, device_obj, torch)
            if not np.isfinite(prediction).all(): raise RuntimeError("nonfinite preflight prediction")
            rows.append({"source_family": source["source_family"], "model": source["model"], "source_lane": source["source_lane"], "target_lane": lane, "status": "pass", "canonical_model_name": identity["canonical_model_name"], "factory_function_identity": identity["factory_function_identity"], "architecture_fingerprint": identity["architecture_fingerprint"]})
    if len(rows) != EXPECTED_PREFLIGHTS: raise RuntimeError(f"expected {EXPECTED_PREFLIGHTS} deduplicated preflights, got {len(rows)}")
    _write_csv(campaign / "batch_preflight_manifest.csv", rows); status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8")); status.update({"status": "evaluation_plan_ready", "preflights": len(rows), "next_gate": "dry_run_or_authorized_submit"}); _write_json(campaign / "campaign_status.json", status); return campaign


def run_shard(campaign_dir: str | Path, shard_id: int, *, repository_root: str | Path = ".", device: str = "cuda:0") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir); import h5py, torch
    status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
    if status["status"] not in {"evaluation_plan_ready", "evaluation_submitted"}: raise RuntimeError("preflight gate not passed")
    selected = [row for row in _read_csv(campaign / "evaluation_plan.csv") if int(row["shard_id"]) == int(shard_id)]
    if not selected: raise RuntimeError("empty shard")
    cache_rows = {row["domain"]: row for row in _read_csv(campaign / "test_cache_manifest.csv")}; out = campaign / "shards" / f"shard_{int(shard_id):02d}"; out.mkdir(parents=True, exist_ok=True); device_obj = torch.device(device); metric_rows = []
    for cell in selected:
        cache = cache_rows[cell["target_domain"]]
        with h5py.File(cache["cache_path"], "r") as data: x, target, k = np.asarray(data["input"], dtype=np.float32), np.asarray(data["Wh"], dtype=np.float32), np.asarray(data["k_points"], dtype=np.float32)
        checkpoint = Path(cell["checkpoint_path"])
        if sha256_file(checkpoint) != cell["checkpoint_sha256"]: raise RuntimeError("checkpoint hash drift")
        payload = _load_checkpoint(checkpoint, device_obj, torch); identity = _identity(payload)
        if identity.get("canonical_model_name") != cell["source_model"]: raise RuntimeError("canonical model name drift")
        started = time.perf_counter(); prediction = _predict(payload, x, k, cell["target_lane"], 64, device_obj, torch); elapsed = time.perf_counter() - started
        metrics, sample_mre, per_band = _metric_rows(prediction, target); np.savez_compressed(out / f"{cell['evaluation_id']}.npz", prediction=prediction, sample_mre=sample_mre, target=target)
        _write_csv(out / f"{cell['evaluation_id']}_per_band.csv", [{"evaluation_id": cell["evaluation_id"], **item} for item in per_band]); metric_rows.append({**cell, **metrics, "status": "completed", "source_normalizer_reused": True, "lane_id": 0 if cell["target_lane"] == "B" else 1, "inference_time_s": elapsed, "throughput_samples_per_s": len(x) / max(elapsed, 1e-12)})
    _write_csv(out / "evaluation_cells.csv", metric_rows); _write_json(out / "results.json", {"status": "shard_success", "shard_id": int(shard_id), "cell_count": len(metric_rows), "nonfinite": False}); return out


def finalize_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir); plan = _read_csv(campaign / "evaluation_plan.csv"); rows: list[dict[str, str]] = []; arrays: dict[str, np.ndarray] = {}
    for directory in sorted((campaign / "shards").glob("shard_*")):
        cell_file = directory / "evaluation_cells.csv"
        if not cell_file.is_file(): continue
        for row in _read_csv(cell_file):
            archive = directory / f"{row['evaluation_id']}.npz"
            if not archive.is_file(): raise RuntimeError(f"missing cell evidence: {archive}")
            rows.append(row); arrays[row["evaluation_id"]] = np.load(archive)["sample_mre"]
    ids = [row["evaluation_id"] for row in rows]
    if len(rows) != EXPECTED_CELLS or set(ids) != {row["evaluation_id"] for row in plan} or len(set(ids)) != EXPECTED_CELLS: raise RuntimeError("510-cell completeness failure")
    if any(row.get("status") != "completed" or not math.isfinite(float(row["bands_2_10_mre_pct"])) for row in rows): raise RuntimeError("failed/nonfinite cell")
    _write_csv(campaign / "evaluation_cells.csv", rows)
    config = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    legacy_root = _resolve(root, str(config.get("legacy_272_campaign", "")))
    legacy_rows_path = legacy_root / "evaluation_cells.csv"
    if not legacy_rows_path.is_file():
        raise RuntimeError(f"legacy 272-cell evidence is required for numeric regression: {legacy_rows_path}")
    legacy_rows = _read_csv(legacy_rows_path)
    current_by_identity = {(r["source_family"], r["source_model"], r["source_lane"], r["source_seed"], r["target_domain"]): r for r in rows if r["source_family"] in {"K20", "F50"} and r["target_family"] in {"K20", "F50"}}
    legacy_by_identity = {(r["source_family"], r["source_model"], r["source_lane"], r["source_seed"], r["target_domain"]): r for r in legacy_rows}
    if len(current_by_identity) != 272 or set(current_by_identity) != set(legacy_by_identity):
        raise RuntimeError("legacy 272-cell identity coverage mismatch")
    tolerance = float(config["statistics"]["legacy_272_abs_tol"]); regression_rows = []
    for identity, current in current_by_identity.items():
        legacy = legacy_by_identity[identity]
        current_npz = campaign / "shards" / f"shard_{int(current['shard_id']):02d}" / f"{current['evaluation_id']}.npz"
        legacy_npz = legacy_root / "shards" / f"shard_{int(legacy['shard_id']):02d}" / f"{legacy['evaluation_id']}.npz"
        if not current_npz.is_file() or not legacy_npz.is_file(): raise RuntimeError(f"legacy numeric evidence missing for {identity}")
        current_prediction = np.load(current_npz)["prediction"]; legacy_prediction = np.load(legacy_npz)["prediction"]
        if current_prediction.shape != legacy_prediction.shape: raise RuntimeError(f"legacy prediction shape mismatch: {identity}")
        delta = float(np.max(np.abs(current_prediction - legacy_prediction)))
        regression_rows.append({"source_family": identity[0], "source_model": identity[1], "source_lane": identity[2], "source_seed": identity[3], "target_domain": identity[4], "max_abs_prediction_delta": delta, "pass": delta <= tolerance})
    _write_csv(campaign / "legacy_272_numeric_regression.csv", regression_rows)
    if not all(row["pass"] for row in regression_rows): raise RuntimeError("legacy 272-cell numerical regression exceeds tolerance")
    for field, selector in (("domain_summary.csv", lambda r: (r["target_domain"], r["comparison_kind"])), ("seed_summary.csv", lambda r: (r["source_family"], r["source_seed"]))):
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows: groups[selector(row)].append(row)
        _write_csv(campaign / field, [{"group": "/".join(key), "cell_count": len(value), "bands_2_10_mre_pct_mean": float(np.mean([float(v["bands_2_10_mre_pct"]) for v in value]))} for key, value in sorted(groups.items())])
    _write_csv(campaign / "per_band_metrics.csv", [{"evaluation_id": r["evaluation_id"], "bands_2_10_mre_pct": r["bands_2_10_mre_pct"]} for r in rows]); _write_csv(campaign / "per_k_metrics.csv", [{"evaluation_id": r["evaluation_id"], "r2": r["bands_2_10_r2_variance_weighted"]} for r in rows])
    _write_json(campaign / "results.json", {"status": "evaluation_complete_pending_report", "completed_checkpoints": EXPECTED_CHECKPOINTS, "completed_cells": EXPECTED_CELLS, "completed_shards": EXPECTED_SHARDS, "legacy_272_numeric_regression": "pass", "nonfinite": False, "surrogate_only": True, "real_mpb_replay": False})
    _write_json(campaign / "campaign_status.json", {"status": "evaluation_complete_pending_report", "planned_cells": EXPECTED_CELLS, "completed_cells": len(rows), "next_gate": "legacy_272_numeric_regression_and_draft_report_review"}); return campaign


def status_campaign(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> dict[str, Any]:
    campaign = _resolve(Path(repository_root).resolve(), campaign_dir); status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8")); plan = _read_csv(campaign / "evaluation_plan.csv") if (campaign / "evaluation_plan.csv").is_file() else []; completed = [r for path in (campaign / "shards").glob("shard_*/evaluation_cells.csv") for r in _read_csv(path)]
    return {**status, "planned_cells": len(plan), "completed_cells": len(completed), "missing_cells": len({r["evaluation_id"] for r in plan} - {r["evaluation_id"] for r in completed})}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "preflight", "run-shard", "finalize", "status"):
        item = sub.add_parser(name); item.add_argument("--campaign-dir", required=True); item.add_argument("--repository-root", default=".")
        if name == "freeze": item.add_argument("--config", default="configs/research/full_cell_k5_six_domain_n40_v1.yaml")
        if name in {"preflight", "run-shard"}: item.add_argument("--device", default="cuda:0")
        if name == "run-shard": item.add_argument("--shard-id", required=True, type=int)
    args = parser.parse_args(argv)
    if args.command == "freeze": print(freeze_campaign(args.campaign_dir, repository_root=args.repository_root, config_path=args.config))
    elif args.command == "preflight": print(run_preflight(args.campaign_dir, repository_root=args.repository_root, device=args.device))
    elif args.command == "run-shard": print(run_shard(args.campaign_dir, args.shard_id, repository_root=args.repository_root, device=args.device))
    elif args.command == "finalize": print(finalize_campaign(args.campaign_dir, repository_root=args.repository_root))
    else: print(json.dumps(status_campaign(args.campaign_dir, repository_root=args.repository_root), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()


__all__ = ["DOMAINS", "EXPECTED_CELLS", "EXPECTED_CHECKPOINTS", "EXPECTED_PREFLIGHTS", "EXPECTED_SHARDS", "build_evaluation_cells", "finalize_campaign", "freeze_campaign", "run_preflight", "run_shard", "status_campaign"]
