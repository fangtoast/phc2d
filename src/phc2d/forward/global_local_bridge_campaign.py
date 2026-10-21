"""Freeze and gate the EXP-FWD-020 six-domain global-local bridge campaign.

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
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.global_local_bridge_evaluation import cell_metrics, mean_std, paired_bootstrap
from phc2d.forward.global_local_bridge_training import (
    STANDARD_TRAINING_ARTIFACTS,
    audit_training_manifest,
    audit_training_task_artifacts,
    run_training_task,
)


VARIANTS = ("baseline", "candidate")


def _sha256(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest cannot be empty")
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("EXP-FWD-020 controller config must be a mapping")
    return dict(value)


def _portable(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _runner_config(controller: Mapping[str, Any], domain: str, variant: str, seed: int, epochs: int) -> dict[str, Any]:
    dataset = dict(controller["datasets"][domain])
    params = {**dict(controller["model"]["shared_params"]), **dict(controller["model"]["paired_variants"][variant])}
    lane = "binary" if domain.endswith("binary") else "graded"
    return {
        "experiment": {"experiment_id": "EXP-FWD-020", "family": "full_cell_global_local_bridge", "lane": lane, "domain": domain, "variant": variant, "evidence_status": "validation_only_g1" if epochs == 20 else "g2_confirmation", "held_out_test_authorized": False},
        "dataset": {"path": dataset.get("n56_path", dataset["path"]), "grid_field": dataset["grid_field"], "expected_dataset_sha256": dataset["sha256"]},
        "representation": {"source": dataset["grid_field"], "sampling": "continuous", "threshold": None, "resolution": 64, "physical_equivalence": True, "label_semantics": f"{domain}_Wh"},
        "models": {controller["model"]["name"]: params},
        "training": {"train_size": 40000 if domain.startswith("f50") else 8000, "train_subset_salt": f"exp-fwd-020-{domain}-v1", "max_epochs": epochs, "required_stop_epoch": 200 if epochs == 200 else None, "early_stopping_patience": 0, "epoch_100_continue_gate": False, "require_terminal_checkpoint": epochs == 200, "convergence_window": [181, 200] if epochs == 200 else None, "convergence_metric": "validation_bands_2_10_mre" if epochs == 200 else None, "convergence_relative_threshold": 0.005 if epochs == 200 else None, "optimizer": controller["training"]["optimizer"], "weight_decay": controller["training"]["weight_decay"], "learning_rates": {controller["model"]["name"]: controller["training"]["learning_rate"]}, "micro_batch_size": controller["training"]["micro_batch_size"], "effective_batch_size": controller["training"]["effective_batch_size"], "loss": controller["training"]["loss"], "checkpoint_selection": "validation_bands_2_10_mre"},
        "runtime": {"device": "cuda:0", "require_accelerator": True, "precision": "fp32", "amp_enabled": False, "tf32_enabled": False, "num_workers": 4, "pin_memory": True, "non_blocking_transfer": True},
        "output": {"runs_dir": "runs"},
    }


def freeze_g1(config_path: str | Path, output_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve()
    controller_path = Path(config_path); controller_path = controller_path if controller_path.is_absolute() else root / controller_path
    out = Path(output_dir); out = out if out.is_absolute() else root / out
    out.mkdir(parents=True, exist_ok=False)
    controller = _load(controller_path)
    if not bool(controller["authorization"]["granted"]):
        raise RuntimeError("remote campaign authorization is not granted")
    derived = out / "derived_configs"; derived.mkdir()
    rows: list[dict[str, Any]] = []
    for index, domain in enumerate(controller["datasets"]):
        for variant in VARIANTS:
            case_id = f"{domain}_{variant}_s0"
            config = derived / f"{case_id}.yaml"
            config.write_text(yaml.safe_dump(_runner_config(controller, domain, variant, 0, 20), sort_keys=False), encoding="utf-8")
            rows.append({"task_index": len(rows), "case_id": case_id, "domain": domain, "lane": "binary" if domain.endswith("binary") else "graded", "variant": variant, "seed": 0, "config": _portable(config, root), "run_dir": _portable(out / "runs" / case_id, root)})
    _write_csv(out / "task_manifest.csv", rows)
    (out / "freeze.json").write_text(json.dumps({"experiment_id": "EXP-FWD-020", "stage": "G1", "test_targets_loaded": False, "task_count": len(rows), "array_limit": 8, "controller_sha256": _sha256(controller_path)}, indent=2) + "\n", encoding="utf-8")
    return out


def freeze_g2(config_path: str | Path, g1_dir: str | Path, output_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    root = Path(repository_root).resolve(); controller = _load(root / config_path)
    g1 = Path(g1_dir); g1 = g1 if g1.is_absolute() else root / g1
    out = Path(output_dir); out = out if out.is_absolute() else root / out; out.mkdir(parents=True, exist_ok=False)
    with (g1 / "task_manifest.csv").open(encoding="utf-8", newline="") as handle: source = list(csv.DictReader(handle))
    by_case = {row["case_id"]: row for row in source}
    passed: list[str] = []
    decisions: dict[str, Any] = {}
    for domain in controller["datasets"]:
        base, candidate = (by_case[f"{domain}_{variant}_s0"] for variant in VARIANTS)
        outcomes = []
        for row in (base, candidate):
            run = root / row["run_dir"]
            required = ("run.log", "run.meta.json", "config.resolved.yaml", "metrics.json", "results.json", "history.csv", "model.pt")
            if any(not (run / name).is_file() for name in required):
                raise RuntimeError(f"G1 evidence incomplete: {row['case_id']}")
            result = json.loads((run / "results.json").read_text(encoding="utf-8")); metric = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if result.get("test_executed") or result.get("selection_split") != "validation":
                raise RuntimeError(f"G1 validation boundary invalid: {row['case_id']}")
            value = float(metric["best_validation_bands_2_10_mre_pct"])
            if not value > 0.0:
                raise RuntimeError(f"G1 metric is not finite positive: {row['case_id']}")
            outcomes.append(value)
        relative = 100.0 * (outcomes[0] - outcomes[1]) / outcomes[0]
        decision = relative >= float(controller["protocol"]["seed0_screen"]["minimum_relative_improvement_pct_each_lane"])
        decisions[domain] = {"baseline_mre_pct": outcomes[0], "candidate_mre_pct": outcomes[1], "relative_improvement_pct": relative, "passed": decision}
        if decision: passed.append(domain)
    derived = out / "derived_configs"; derived.mkdir(); rows: list[dict[str, Any]] = []
    for domain in passed:
        for variant in VARIANTS:
            for seed in (0, 1, 2):
                case_id = f"{domain}_{variant}_s{seed}"; config = derived / f"{case_id}.yaml"
                config.write_text(yaml.safe_dump(_runner_config(controller, domain, variant, seed, 200), sort_keys=False), encoding="utf-8")
                rows.append({"task_index": len(rows), "case_id": case_id, "domain": domain, "lane": "binary" if domain.endswith("binary") else "graded", "variant": variant, "seed": seed, "config": _portable(config, root), "run_dir": _portable(out / "runs" / case_id, root)})
    if rows: _write_csv(out / "task_manifest.csv", rows)
    else: (out / "task_manifest.csv").write_text("task_index,case_id,domain,lane,variant,seed,config,run_dir\n", encoding="utf-8")
    (out / "candidate_gate.json").write_text(json.dumps({"experiment_id": "EXP-FWD-020", "stage": "G2", "test_targets_loaded": False, "decisions": decisions, "passed_domains": passed, "task_count": len(rows)}, indent=2) + "\n", encoding="utf-8")
    return out


def freeze_exp_fwd_020_continuations(
    source_campaign: str | Path,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ".",
) -> Path:
    """Freeze exactly the 29 non-converged G2 runs for fixed epoch-300 continuation.

    This is deliberately a derivative campaign: source epoch-200 evidence is
    read-only, each warm start records both original checkpoint hashes, and
    no held-out target is opened during freeze.
    """
    root = Path(repository_root).resolve()
    source = _resolve(root, source_campaign)
    out = _resolve(root, output_dir)
    if out.exists():
        raise FileExistsError(f"continuation directory already exists: {out}")
    manifest = source / "task_manifest.csv"
    gate_path = source / "candidate_gate.json"
    if not manifest.is_file() or not gate_path.is_file():
        raise RuntimeError("EXP-FWD-020 G2 source manifest or gate is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rows = _read_csv(manifest)
    identities = {(row["domain"], row["variant"], int(row["seed"])) for row in rows}
    selected_domains = gate.get("selected_domains", gate.get("passed_domains", ()))
    expected = {(domain, variant, seed) for domain in selected_domains for variant in VARIANTS for seed in (0, 1, 2)}
    if len(rows) != 36 or identities != expected:
        raise RuntimeError("EXP-FWD-020 G2 source is not the frozen 36-run matrix")
    out.mkdir(parents=True)
    derived = out / "derived_configs"
    derived.mkdir()
    continuation_rows: list[dict[str, Any]] = []
    untouched: list[dict[str, Any]] = []
    for row in rows:
        source_run = _resolve(root, row["run_dir"])
        required = ("run.log", "run.meta.json", "config.resolved.yaml", "metrics.json", "results.json", "history.csv", "model.pt", "epoch200.pt")
        missing = [name for name in required if not (source_run / name).is_file()]
        if missing:
            raise RuntimeError(f"G2 source artifacts missing for {row['case_id']}: {missing}")
        result = json.loads((source_run / "results.json").read_text(encoding="utf-8"))
        if result.get("test_executed") or result.get("selection_split") != "validation":
            raise RuntimeError(f"G2 validation boundary invalid: {row['case_id']}")
        if int(result.get("stop_epoch", -1)) != 200 or int(result.get("history_epoch_count", -1)) != 200:
            raise RuntimeError(f"G2 source is not a complete epoch-200 run: {row['case_id']}")
        best_hash, terminal_hash = _sha256(source_run / "model.pt"), _sha256(source_run / "epoch200.pt")
        if result.get("checkpoint_sha256") != best_hash or result.get("terminal_checkpoint_sha256") != terminal_hash:
            raise RuntimeError(f"G2 checkpoint provenance mismatch: {row['case_id']}")
        common = {
            **row,
            "source_run": str(source_run),
            "source_best_checkpoint_sha256": best_hash,
            "source_epoch200_checkpoint_sha256": terminal_hash,
            "source_status": result.get("status"),
        }
        if result.get("status") == "formal_training_success":
            untouched.append(common)
            continue
        if result.get("status") != "training_complete_convergence_not_met":
            raise RuntimeError(f"unsupported G2 source status for {row['case_id']}: {result.get('status')}")
        source_config = _resolve(root, row["config"])
        if not source_config.is_file():
            raise FileNotFoundError(source_config)
        config = _load(source_config)
        config["experiment"] = {
            **dict(config["experiment"]),
            "experiment_id": "EXP-FWD-020",
            "evidence_status": "exploratory_validation_only_epoch300_continuation",
            "held_out_test_authorized": False,
        }
        config["training"] = {
            **dict(config["training"]),
            "max_epochs": 300,
            "required_stop_epoch": 300,
            "convergence_window": [281, 300],
            "require_terminal_checkpoint": True,
            "early_stopping_patience": 0,
        }
        config["continuation"] = {
            "mode": "epoch200_weights_optimizer_reset",
            "source_experiment": "EXP-FWD-020",
            "source_epoch": 200,
            "optimizer_reset": True,
            "data_order_state_restored": False,
            "check_interval": 20,
            "relative_threshold": 0.005,
            "max_total_epoch": 300,
            "stop_when_converged": False,
            "source_run": str(source_run),
        }
        config_path = derived / f"{row['case_id']}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        continuation_rows.append({
            **common,
            "task_index": len(continuation_rows),
            "source_config": str(source_config),
            "config": str(config_path),
            "run_dir": str(out / "runs" / row["case_id"]),
        })
    if len(continuation_rows) != 29 or len(untouched) != 7:
        raise RuntimeError("EXP-FWD-020 continuation source must split into 29 non-converged and 7 converged runs")
    _write_csv(out / "task_manifest.csv", continuation_rows)
    _write_csv(out / "untouched_checkpoint_sources.csv", untouched)
    (out / "freeze.json").write_text(json.dumps({
        "experiment_id": "EXP-FWD-020", "stage": "epoch200_to_epoch300_continuation",
        "task_count": 29, "untouched_checkpoint_count": 7, "test_targets_loaded": False,
        "optimizer_state_restored": False, "source_epoch": 200, "terminal_epoch": 300,
        "source_campaign": str(source), "source_manifest_sha256": _sha256(manifest),
    }, indent=2) + "\n", encoding="utf-8")
    return out


def _g3_relationship(source_domain: str, target_domain: str) -> str:
    source_dataset, source_lane = source_domain.split("_", 1)
    target_dataset, target_lane = target_domain.split("_", 1)
    if source_dataset == target_dataset and source_lane == target_lane:
        return "native"
    if source_dataset == target_dataset:
        return "cross_lane"
    if source_lane == target_lane:
        return "cross_dataset"
    return "cross_dataset_cross_lane"


def freeze_g3(
    config_path: str | Path,
    g2_campaign: str | Path,
    continuation_campaign: str | Path,
    output_dir: str | Path,
    *,
    repository_root: str | Path = ".",
    reuse_test_cache_from: str | Path | None = None,
) -> Path:
    """Freeze the authorized one-read EXP-FWD-020 exploratory 36x6 matrix."""
    root = Path(repository_root).resolve()
    config = _load(_resolve(root, config_path))
    if not bool(config.get("authorization", {}).get("held_out_test_authorized")):
        raise RuntimeError("G3 held-out authorization is not enabled")
    if config.get("experiment", {}).get("evidence_status") != "exploratory_held_out_surrogate_only":
        raise RuntimeError("G3 must retain the exploratory surrogate boundary")
    g2 = _resolve(root, g2_campaign)
    continuation = _resolve(root, continuation_campaign)
    out = _resolve(root, output_dir)
    cache_source = _resolve(root, reuse_test_cache_from) if reuse_test_cache_from else None
    if out.exists():
        raise FileExistsError(f"G3 campaign directory already exists: {out}")
    if cache_source is not None:
        source_status = json.loads((cache_source / "campaign_status.json").read_text(encoding="utf-8"))
        if source_status.get("status") != "evaluation_plan_ready" or not source_status.get("test_cache_created_once"):
            raise RuntimeError("G3 cache reuse requires a successful one-read preflight source")
        if not (cache_source / "dataset_manifest.csv").is_file() or not (cache_source / "test_cache").is_dir():
            raise RuntimeError("G3 cache reuse source artifacts are incomplete")
    g2_rows = _read_csv(g2 / "task_manifest.csv")
    continuation_rows = _read_csv(continuation / "task_manifest.csv")
    untouched_rows = _read_csv(continuation / "untouched_checkpoint_sources.csv")
    if len(g2_rows) != 36 or len(continuation_rows) != 29 or len(untouched_rows) != 7:
        raise RuntimeError("G3 requires the frozen 36=29+7 final-checkpoint source inventory")
    final_rows: list[dict[str, Any]] = []
    for row in untouched_rows:
        source_run = _resolve(root, row["source_run"])
        result = json.loads((source_run / "results.json").read_text(encoding="utf-8"))
        checkpoint = source_run / "model.pt"
        required = list(STANDARD_TRAINING_ARTIFACTS) + ["epoch200.pt"]
        missing = [name for name in required if not (source_run / name).is_file() or not (source_run / name).stat().st_size]
        if missing or result.get("status") != "formal_training_success" or result.get("test_executed"):
            raise RuntimeError(f"untouched G2 source invalid: {row['case_id']}")
        final_rows.append({**row, "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "checkpoint_origin": "g2_validation_best", "training_convergence_status": result["status"]})
    for row in continuation_rows:
        run = _resolve(root, row["run_dir"])
        result_path, checkpoint = run / "results.json", run / "model.pt"
        if not result_path.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"continuation result/checkpoint missing: {row['case_id']}")
        audit_training_task_artifacts({**row, "_run_dir": str(run), "_is_continuation": "True"}, admission_probe=False)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("test_executed") or int(result.get("stop_epoch", -1)) != 300 or int(result.get("history_epoch_count", -1)) != 300:
            raise RuntimeError(f"continuation final state invalid: {row['case_id']}")
        final_rows.append({**row, "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "checkpoint_origin": "epoch300_validation_best", "training_convergence_status": result["status"]})
    identities = {(row["domain"], row["variant"], int(row["seed"])) for row in final_rows}
    expected = {(row["domain"], row["variant"], int(row["seed"])) for row in g2_rows}
    if len(final_rows) != 36 or identities != expected:
        raise RuntimeError("final checkpoint inventory has a collision or omission")
    out.mkdir(parents=True)
    datasets: list[dict[str, Any]] = []
    for domain, contract in config["datasets"].items():
        path = _resolve(root, str(contract["path"]))
        if _sha256(path) != str(contract["sha256"]):
            raise RuntimeError(f"dataset hash drift: {domain}")
        # The cache is deliberately absent at freeze time: the single-card validation
        # preflight must pass before this controller is allowed to read idx_test.
        datasets.append({"domain": domain, "path": str(path), "grid_field": contract["grid_field"], "dataset_sha256": contract["sha256"], "test_size": int(contract["test_size"]), "cache_path": str(out / "test_cache" / f"{domain}.h5"), "cache_sha256": "", "idx_test_sha256": "", "k_points_sha256": ""})
    cells: list[dict[str, Any]] = []
    for checkpoint in sorted(final_rows, key=lambda item: (item["domain"], item["variant"], int(item["seed"]))):
        for target in config["datasets"]:
            cells.append({"evaluation_id": f"g3_{len(cells):03d}", "checkpoint_id": checkpoint["case_id"], "source_domain": checkpoint["domain"], "source_lane": checkpoint["lane"], "source_variant": checkpoint["variant"], "source_seed": checkpoint["seed"], "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"], "target_domain": target, "target_lane": "binary" if target.endswith("binary") else "graded", "relationship": _g3_relationship(checkpoint["domain"], target), "shard_id": len(cells) % 8, "status": "planned"})
    relation_counts = {name: sum(row["relationship"] == name for row in cells) for name in ("native", "cross_lane", "cross_dataset", "cross_dataset_cross_lane")}
    if len(cells) != 216 or relation_counts != dict(config["matrix"]["relationship_counts"]):
        raise RuntimeError(f"G3 matrix contract mismatch: {relation_counts}")
    _write_csv(out / "checkpoint_manifest.csv", final_rows)
    _write_csv(out / "dataset_manifest.csv", datasets)
    _write_csv(out / "evaluation_plan.csv", cells)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "freeze.json").write_text(json.dumps({"experiment_id": "EXP-FWD-020", "stage": "G3", "test_cache_created_once": False, "test_cache_gate": "g3_six_domain_preflight", "test_cache_source_campaign": str(cache_source) if cache_source is not None else "", "checkpoint_count": 36, "cell_count": 216, "relationship_counts": relation_counts, "plan_sha256": _sha256(out / "evaluation_plan.csv")}, indent=2) + "\n", encoding="utf-8")
    (out / "campaign_status.json").write_text(json.dumps({"status": "frozen_preflight_required", "planned_cells": 216, "completed_cells": 0, "next_gate": "g3_six_domain_preflight"}, indent=2) + "\n", encoding="utf-8")
    return out


def _g3_checkpoint(path: Path, device: Any, torch: Any) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping) or not {"state_dict", "model_name", "model_params", "target_mean", "target_std"}.issubset(payload):
        raise RuntimeError(f"checkpoint payload is incomplete: {path}")
    return dict(payload)


def _g3_predict(payload: Mapping[str, Any], x: np.ndarray, k_points: np.ndarray, batch: int, device: Any, torch: Any) -> np.ndarray:
    from phc2d.models import build_model, get_model_spec

    model = build_model(str(payload["model_name"]), input_shape=tuple(payload["input_shape"]), output_shape=tuple(payload["output_shape"]), **dict(payload["model_params"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    spec = get_model_spec(str(payload["model_name"]))
    mean = np.asarray(payload["target_mean"].detach().cpu() if hasattr(payload["target_mean"], "detach") else payload["target_mean"], dtype=np.float32)
    std = np.asarray(payload["target_std"].detach().cpu() if hasattr(payload["target_std"], "detach") else payload["target_std"], dtype=np.float32)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for begin in range(0, len(x), batch):
            xb = torch.from_numpy(x[begin:begin + batch]).to(device)
            kwargs = {"k_points": torch.from_numpy(k_points).to(device)} if spec.accepts_k_points else {}
            parts.append((model(xb, **kwargs).detach().cpu().numpy() * std + mean).astype(np.float32))
    return np.concatenate(parts, axis=0)


def _g3_reuse_test_cache(source_campaign: Path, datasets: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Validate and reference an already authorized, one-read six-domain cache."""
    status = json.loads((source_campaign / "campaign_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "evaluation_plan_ready" or not status.get("test_cache_created_once"):
        raise RuntimeError("G3 cache reuse source has not passed one-read preflight")
    cached = {row["domain"]: row for row in _read_csv(source_campaign / "dataset_manifest.csv")}
    if set(cached) != {row["domain"] for row in datasets}:
        raise RuntimeError("G3 cache reuse source domains do not match frozen datasets")
    resolved: list[dict[str, Any]] = []
    for dataset in datasets:
        source = cached[dataset["domain"]]
        if source.get("dataset_sha256") != dataset["dataset_sha256"] or not source.get("cache_sha256"):
            raise RuntimeError(f"G3 cache reuse provenance mismatch: {dataset['domain']}")
        cache = Path(source["cache_path"])
        if not cache.is_file() or _sha256(cache) != source["cache_sha256"]:
            raise RuntimeError(f"G3 cache reuse hash drift: {dataset['domain']}")
        with __import__("h5py").File(cache, "r") as handle:
            required = {"input", "Wh", "k_points", "test_indices"}
            missing = required.difference(handle.keys())
            if missing or len(handle["input"]) != int(dataset["test_size"]) or len(handle["Wh"]) != int(dataset["test_size"]):
                raise RuntimeError(f"G3 cache reuse schema mismatch: {dataset['domain']}")
        resolved.append({**dataset, "cache_path": str(cache), "cache_sha256": source["cache_sha256"], "idx_test_sha256": source["idx_test_sha256"], "k_points_sha256": source["k_points_sha256"], "test_cache_origin": "reused_one_read_cache"})
    return resolved


def preflight_g3(campaign_dir: str | Path, *, repository_root: str | Path = ".", device: str = "cuda:0") -> Path:
    """Prove six-domain validation inference before the one permitted test-cache read."""
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir)
    status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "frozen_preflight_required":
        raise RuntimeError(f"G3 preflight is not available: {status.get('status')}")
    import h5py
    import torch

    checkpoints = _read_csv(campaign / "checkpoint_manifest.csv")
    datasets = _read_csv(campaign / "dataset_manifest.csv")
    if len(checkpoints) != 36 or len(datasets) != 6:
        raise RuntimeError("G3 preflight requires frozen 36 checkpoints and six datasets")
    device_obj = torch.device(device)
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        path = Path(checkpoint["checkpoint_path"])
        if _sha256(path) != checkpoint["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash drift: {path}")
        payloads.append(_g3_checkpoint(path, device_obj, torch))
        rows.append({"kind": "checkpoint_load", "checkpoint_id": checkpoint["case_id"], "status": "passed", "checkpoint_sha256": checkpoint["checkpoint_sha256"]})
    for position, dataset in enumerate(datasets):
        path = Path(dataset["path"])
        if _sha256(path) != dataset["dataset_sha256"]:
            raise RuntimeError(f"dataset hash drift: {dataset['domain']}")
        with h5py.File(path, "r") as source:
            required = {dataset["grid_field"], "Wh", "idx_val", "k_points"}
            missing = required.difference(source.keys())
            if missing:
                raise RuntimeError(f"validation fields missing for {dataset['domain']}: {sorted(missing)}")
            indices = np.asarray(source["idx_val"], dtype=np.int64)
            if len(indices) < 2:
                raise RuntimeError(f"validation split too small for batch check: {dataset['domain']}")
            check = indices[: min(4, len(indices))]
            x = np.asarray(source[dataset["grid_field"]][check], dtype=np.float32)
            k_points = np.asarray(source["k_points"], dtype=np.float32)
        prediction_b1 = _g3_predict(payloads[position], x, k_points, 1, device_obj, torch)
        prediction_bn = _g3_predict(payloads[position], x, k_points, len(x), device_obj, torch)
        if not np.isfinite(prediction_b1).all() or not np.allclose(prediction_b1, prediction_bn, rtol=1.0e-5, atol=1.0e-6):
            raise RuntimeError(f"validation forward/batch consistency failed: {dataset['domain']}")
        rows.append({"kind": "validation_forward", "target_domain": dataset["domain"], "checkpoint_id": checkpoints[position]["case_id"], "samples": len(x), "status": "passed", "batch_consistent": True})
    freeze = json.loads((campaign / "freeze.json").read_text(encoding="utf-8"))
    source_value = str(freeze.get("test_cache_source_campaign", ""))
    if source_value:
        cache_source = _resolve(root, source_value)
        resolved_datasets = _g3_reuse_test_cache(cache_source, datasets)
        (campaign / "test_cache").symlink_to(cache_source / "test_cache", target_is_directory=True)
        rows.extend({"kind": "test_cache_reuse", "target_domain": row["domain"], "status": "passed", "cache_path": row["cache_path"], "cache_sha256": row["cache_sha256"]} for row in resolved_datasets)
    else:
        # Only the successful preflight reaches this point.  idx_test is read once per
        # frozen target and immediately copied into an immutable evaluation cache.
        cache_dir = campaign / "test_cache"; cache_dir.mkdir(exist_ok=False)
        resolved_datasets: list[dict[str, Any]] = []
        for dataset in datasets:
            path = Path(dataset["path"])
            with h5py.File(path, "r") as source:
                required = {dataset["grid_field"], "Wh", "idx_test", "k_points"}
                missing = required.difference(source.keys())
                if missing:
                    raise RuntimeError(f"test-cache fields missing for {dataset['domain']}: {sorted(missing)}")
                indices = np.asarray(source["idx_test"], dtype=np.int64)
                if len(indices) != int(dataset["test_size"]):
                    raise RuntimeError(f"test size drift: {dataset['domain']}")
                cache = Path(dataset["cache_path"])
                with h5py.File(cache, "w") as destination:
                    destination.create_dataset("input", data=source[dataset["grid_field"]][indices], dtype="f4")
                    destination.create_dataset("Wh", data=source["Wh"][indices], dtype="f4")
                    destination.create_dataset("k_points", data=source["k_points"], dtype="f4")
                    destination.create_dataset("test_indices", data=indices, dtype="i8")
                resolved_datasets.append({**dataset, "idx_test_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "k_points_sha256": hashlib.sha256(np.asarray(source["k_points"], dtype=np.float32).tobytes()).hexdigest(), "cache_sha256": _sha256(cache)})
    _write_csv(campaign / "dataset_manifest.csv", resolved_datasets)
    _write_csv(campaign / "preflight_manifest.csv", rows)
    (campaign / "campaign_status.json").write_text(json.dumps({"status": "evaluation_plan_ready", "planned_cells": 216, "completed_cells": 0, "preflight_checkpoint_loads": 36, "preflight_validation_forwards": 6, "test_cache_created_once": True, "test_cache_original_reads_this_campaign": 0 if source_value else 6, "test_cache_source_campaign": source_value, "next_gate": "authorized_g3_array_submission"}, indent=2) + "\n", encoding="utf-8")
    return campaign


def run_g3_shard(campaign_dir: str | Path, shard_id: int, *, repository_root: str | Path = ".", device: str = "cuda:0") -> Path:
    root = Path(repository_root).resolve(); campaign = _resolve(root, campaign_dir)
    status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"evaluation_plan_ready", "evaluation_submitted"}:
        raise RuntimeError(f"G3 is not runnable: {status.get('status')}")
    import h5py
    import torch

    selected = [row for row in _read_csv(campaign / "evaluation_plan.csv") if int(row["shard_id"]) == int(shard_id)]
    if not selected:
        raise RuntimeError(f"empty G3 shard: {shard_id}")
    caches = {row["domain"]: row for row in _read_csv(campaign / "dataset_manifest.csv")}
    out = campaign / "shards" / f"shard_{int(shard_id):02d}"; out.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for cell in selected:
        checkpoint = Path(cell["checkpoint_path"])
        if _sha256(checkpoint) != cell["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash drift: {checkpoint}")
        cache = caches[cell["target_domain"]]
        if _sha256(cache["cache_path"]) != cache["cache_sha256"]:
            raise RuntimeError(f"G3 cache hash drift: {cell['target_domain']}")
        with h5py.File(cache["cache_path"], "r") as handle:
            x, target, k_points = np.asarray(handle["input"], dtype=np.float32), np.asarray(handle["Wh"], dtype=np.float32), np.asarray(handle["k_points"], dtype=np.float32)
        device_obj = torch.device(device); torch.cuda.reset_peak_memory_stats()
        payload = _g3_checkpoint(checkpoint, device_obj, torch)
        started = time.perf_counter(); prediction = _g3_predict(payload, x, k_points, 64, device_obj, torch); elapsed = time.perf_counter() - started
        metrics, per_band, per_k, sample_arrays = cell_metrics(prediction, target)
        np.savez_compressed(out / f"{cell['evaluation_id']}.npz", **sample_arrays)
        _write_csv(out / f"{cell['evaluation_id']}_per_band.csv", [{"evaluation_id": cell["evaluation_id"], **value} for value in per_band])
        _write_csv(out / f"{cell['evaluation_id']}_per_k.csv", [{"evaluation_id": cell["evaluation_id"], **value} for value in per_k])
        _write_csv(out / f"{cell['evaluation_id']}_gap_decision.csv", [{"evaluation_id": cell["evaluation_id"], **{key: value for key, value in metrics.items() if "gap" in key or key.startswith("selected_") or key.startswith("upper_") or key.startswith("band_k_") or key.startswith("complete_") or key.startswith("pseudo_")}}])
        rows.append({**cell, **metrics, "status": "completed", "source_normalizer_reused": True, "actual_batch_size": min(64, len(x)), "inference_time_s": elapsed, "latency_per_sample_s": elapsed / max(len(x), 1), "throughput_samples_per_s": len(x) / max(elapsed, 1.0e-12), "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2)})
    _write_csv(out / "evaluation_cells.csv", rows)
    (out / "results.json").write_text(json.dumps({"status": "shard_success", "cell_count": len(rows)}, indent=2) + "\n", encoding="utf-8")
    return out


def status_g3(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> dict[str, Any]:
    campaign = _resolve(Path(repository_root).resolve(), campaign_dir)
    status = json.loads((campaign / "campaign_status.json").read_text(encoding="utf-8"))
    plan = _read_csv(campaign / "evaluation_plan.csv")
    complete = [row for shard in (campaign / "shards").glob("shard_*") if (shard / "evaluation_cells.csv").is_file() for row in _read_csv(shard / "evaluation_cells.csv")]
    summary = {**status, "planned_cells": len(plan), "completed_cells": len(complete), "missing_cells": len({row["evaluation_id"] for row in plan} - {row["evaluation_id"] for row in complete})}
    (campaign / "status_snapshot.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _g3_archive(campaign: Path, row: Mapping[str, str]) -> Path:
    return campaign / "shards" / f"shard_{int(row['shard_id']):02d}" / f"{row['evaluation_id']}.npz"


def _g3_sample_mre(campaign: Path, row: Mapping[str, str]) -> np.ndarray:
    with np.load(_g3_archive(campaign, row)) as archive:
        return np.asarray(archive["sample_mre_bands_2_10_pct"], dtype=np.float64)


def _g3_metric_summary(rows: list[dict[str, str]], **identity: Any) -> dict[str, Any]:
    mean, std = mean_std(rows)
    return {**identity, "cell_count": len(rows), "bands_2_10_mre_pct_mean": mean, "bands_2_10_mre_pct_std": std}


def _g3_collect_cell_artifact_rows(campaign: Path, rows: list[dict[str, str]], suffix: str) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []
    for row in rows:
        path = _g3_archive(campaign, row).with_name(f"{row['evaluation_id']}_{suffix}.csv")
        if not path.is_file():
            raise RuntimeError(f"G3 {suffix} artifact missing: {path}")
        collected.extend(_read_csv(path))
    return collected


def finalize_g3(campaign_dir: str | Path, *, repository_root: str | Path = ".") -> Path:
    """Fail closed on incomplete G3 evidence, then publish exploratory summaries."""
    campaign = _resolve(Path(repository_root).resolve(), campaign_dir)
    plan = _read_csv(campaign / "evaluation_plan.csv")
    rows = [row for shard in (campaign / "shards").glob("shard_*") if (shard / "evaluation_cells.csv").is_file() for row in _read_csv(shard / "evaluation_cells.csv")]
    expected = {row["evaluation_id"] for row in plan}
    observed = [row["evaluation_id"] for row in rows]
    if len(plan) != 216 or len(rows) != 216 or set(observed) != expected or len(set(observed)) != 216:
        raise RuntimeError("G3 finalizer requires exactly 216 unique completed cells")
    if any(row.get("status") != "completed" or not np.isfinite(float(row["bands_2_10_mre_pct"])) for row in rows):
        raise RuntimeError("G3 finalizer found nonfinite or incomplete metrics")
    _write_csv(campaign / "evaluation_cells.csv", rows)
    relationship_summary: list[dict[str, Any]] = []
    for relation in ("native", "cross_lane", "cross_dataset", "cross_dataset_cross_lane"):
        relationship_summary.append(_g3_metric_summary([row for row in rows if row["relationship"] == relation], relationship=relation))
    _write_csv(campaign / "relationship_summary.csv", relationship_summary)

    source_target_summary = [
        _g3_metric_summary(
            selected,
            source_domain=source,
            target_domain=target,
            relationship=selected[0]["relationship"],
        )
        for source in sorted({row["source_domain"] for row in rows})
        for target in sorted({row["target_domain"] for row in rows})
        if (selected := [row for row in rows if row["source_domain"] == source and row["target_domain"] == target])
    ]
    if len(source_target_summary) != 36:
        raise RuntimeError("G3 source-target summary must contain all 36 directed relationships")
    _write_csv(campaign / "source_target_summary.csv", source_target_summary)

    dataset_transfer_summary = []
    for source_dataset in ("k5", "k20", "f50"):
        for target_dataset in ("k5", "k20", "f50"):
            if source_dataset == target_dataset:
                continue
            selected = [row for row in rows if row["source_domain"].split("_", 1)[0] == source_dataset and row["target_domain"].split("_", 1)[0] == target_dataset]
            dataset_transfer_summary.append(_g3_metric_summary(selected, source_dataset=source_dataset, target_dataset=target_dataset))
    _write_csv(campaign / "dataset_transfer_summary.csv", dataset_transfer_summary)

    lane_transfer_summary = []
    for direction in ("binary_to_graded", "graded_to_binary"):
        source_lane, target_lane = direction.split("_to_")
        for dataset_changed in (False, True):
            selected = [row for row in rows if row["source_lane"] == source_lane and row["target_lane"] == target_lane and (row["source_domain"].split("_", 1)[0] != row["target_domain"].split("_", 1)[0]) == dataset_changed]
            lane_transfer_summary.append(_g3_metric_summary(selected, direction=direction, dataset_changed=dataset_changed))
    _write_csv(campaign / "lane_transfer_summary.csv", lane_transfer_summary)

    seed_summary = []
    for source in sorted({row["source_domain"] for row in rows}):
        for variant in VARIANTS:
            for target in sorted({row["target_domain"] for row in rows}):
                selected = [row for row in rows if row["source_domain"] == source and row["source_variant"] == variant and row["target_domain"] == target]
                if len(selected) != 3:
                    raise RuntimeError("G3 seed summary requires exactly three seeds per source/variant/target")
                summary = _g3_metric_summary(selected, source_domain=source, source_variant=variant, target_domain=target)
                summary["seed0_mre_pct"], summary["seed1_mre_pct"], summary["seed2_mre_pct"] = (float(next(row["bands_2_10_mre_pct"] for row in selected if row["source_seed"] == str(seed))) for seed in (0, 1, 2))
                seed_summary.append(summary)
    _write_csv(campaign / "seed_summary.csv", seed_summary)

    native = {row["checkpoint_id"]: row for row in rows if row["relationship"] == "native"}
    if len(native) != 36:
        raise RuntimeError("G3 must have one native reference per frozen checkpoint")
    degradation: list[dict[str, Any]] = []
    for row in rows:
        if row["relationship"] == "native":
            continue
        reference = native.get(row["checkpoint_id"])
        if reference is None:
            raise RuntimeError(f"native reference missing for {row['checkpoint_id']}")
        native_mre, cross_mre = float(reference["bands_2_10_mre_pct"]), float(row["bands_2_10_mre_pct"])
        degradation.append({
            "checkpoint_id": row["checkpoint_id"], "evaluation_id": row["evaluation_id"], "native_evaluation_id": reference["evaluation_id"], "source_domain": row["source_domain"], "target_domain": row["target_domain"], "relationship": row["relationship"],
            "native_mre_pct": native_mre, "cross_mre_pct": cross_mre, "mre_delta_pp": cross_mre - native_mre, "cross_native_ratio": cross_mre / max(native_mre, 1.0e-12), "relative_degradation_pct": (cross_mre - native_mre) / max(native_mre, 1.0e-12) * 100.0,
        })
    if len(degradation) != 180:
        raise RuntimeError("G3 requires exactly 180 checkpoint degradation rows")
    _write_csv(campaign / "checkpoint_degradation.csv", degradation)

    degradation_summary = []
    for checkpoint_id in sorted(native):
        selected = [row for row in degradation if row["checkpoint_id"] == checkpoint_id]
        result: dict[str, Any] = {"checkpoint_id": checkpoint_id, "native_mre_pct": float(native[checkpoint_id]["bands_2_10_mre_pct"])}
        for relationship in ("cross_lane", "cross_dataset", "cross_dataset_cross_lane"):
            values = [float(row["relative_degradation_pct"]) for row in selected if row["relationship"] == relationship]
            result[f"{relationship}_count"] = len(values)
            result[f"{relationship}_mean_relative_degradation_pct"] = float(np.mean(values))
            result[f"{relationship}_worst_relative_degradation_pct"] = float(np.max(values))
        degradation_summary.append(result)
    _write_csv(campaign / "checkpoint_degradation_summary.csv", degradation_summary)

    kernel_bootstrap: list[dict[str, Any]] = []
    for source in sorted({row["source_domain"] for row in rows}):
        for target in sorted({row["target_domain"] for row in rows}):
            for seed in ("0", "1", "2"):
                pair = [row for row in rows if row["source_domain"] == source and row["target_domain"] == target and row["source_seed"] == seed]
                if len(pair) != 2 or {row["source_variant"] for row in pair} != set(VARIANTS):
                    raise RuntimeError("G3 kernel bootstrap requires one baseline/candidate pair per source/target/seed")
                by_variant = {row["source_variant"]: row for row in pair}
                comparison = paired_bootstrap(_g3_sample_mre(campaign, by_variant["candidate"]), _g3_sample_mre(campaign, by_variant["baseline"]), key=f"{source}:{target}:{seed}")
                kernel_bootstrap.append({"source_domain": source, "target_domain": target, "seed": seed, "relationship": pair[0]["relationship"], "baseline_evaluation_id": by_variant["baseline"]["evaluation_id"], "candidate_evaluation_id": by_variant["candidate"]["evaluation_id"], **comparison})
    if len(kernel_bootstrap) != 108:
        raise RuntimeError("G3 requires exactly 108 paired kernel bootstrap comparisons")
    _write_csv(campaign / "kernel_paired_bootstrap.csv", kernel_bootstrap)

    kernel_direction_summary = []
    for source in sorted({row["source_domain"] for row in rows}):
        for target in sorted({row["target_domain"] for row in rows}):
            selected = [row for row in kernel_bootstrap if row["source_domain"] == source and row["target_domain"] == target]
            deltas = [float(row["mre_delta_pp"]) for row in selected]
            conclusions = [row["conclusion"] for row in selected]
            kernel_direction_summary.append({"source_domain": source, "target_domain": target, "seed0_mre_delta_pp": deltas[0], "seed1_mre_delta_pp": deltas[1], "seed2_mre_delta_pp": deltas[2], "improved_seed_count": sum(delta < 0.0 for delta in deltas), "regressed_seed_count": sum(delta > 0.0 for delta in deltas), "direction_consistent": bool(all(delta < 0.0 for delta in deltas) or all(delta > 0.0 for delta in deltas)), "mean_mre_delta_pp": float(np.mean(deltas)), "mre_delta_pp_std": float(np.std(deltas, ddof=0)), "bootstrap_conclusion_consistent": len(set(conclusions)) == 1, "bootstrap_conclusion": conclusions[0] if len(set(conclusions)) == 1 else "mixed"})
    _write_csv(campaign / "kernel_direction_summary.csv", kernel_direction_summary)

    per_band = _g3_collect_cell_artifact_rows(campaign, rows, "per_band")
    per_k = _g3_collect_cell_artifact_rows(campaign, rows, "per_k")
    gaps = _g3_collect_cell_artifact_rows(campaign, rows, "gap_decision")
    cell_lookup = {row["evaluation_id"]: row for row in rows}
    _write_csv(campaign / "per_band_metrics.csv", [{**cell_lookup[row["evaluation_id"]], **row} for row in per_band])
    _write_csv(campaign / "per_k_metrics.csv", [{**cell_lookup[row["evaluation_id"]], **row} for row in per_k])
    _write_csv(campaign / "gap_decision_metrics.csv", [{**cell_lookup[row["evaluation_id"]], **row} for row in gaps])

    resource_summary = []
    for target in sorted({row["target_domain"] for row in rows}):
        selected = [row for row in rows if row["target_domain"] == target]
        resource_summary.append({"target_domain": target, "cell_count": len(selected), "inference_time_total_s": float(sum(float(row["inference_time_s"]) for row in selected)), "latency_per_sample_s_mean": float(np.mean([float(row["latency_per_sample_s"]) for row in selected])), "throughput_samples_per_s_mean": float(np.mean([float(row["throughput_samples_per_s"]) for row in selected])), "peak_gpu_memory_mb_max": float(np.max([float(row["peak_gpu_memory_mb"]) for row in selected])), "actual_batch_size": int(selected[0]["actual_batch_size"])})
    _write_csv(campaign / "resource_summary.csv", resource_summary)

    metadata = {"experiment_id": "EXP-FWD-020", "stage": "G3", "execution_start_git_commit": os.environ.get("PHC2D_EXPECTED_COMMIT", "unknown"), "plan_sha256": _sha256(campaign / "evaluation_plan.csv"), "checkpoint_manifest_sha256": _sha256(campaign / "checkpoint_manifest.csv"), "dataset_manifest_sha256": _sha256(campaign / "dataset_manifest.csv"), "scientific_boundary": "exploratory_held_out_surrogate_only"}
    (campaign / "run.meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (campaign / "run.log").write_text("G3 finalizer completed 216/216 finite surrogate-only cells; human review required.\n", encoding="utf-8")
    _write_csv(campaign / "history.csv", [{"phase": "g3_evaluation", "status": "completed", "note": "No training history: frozen-checkpoint inference only."}])
    (campaign / "metrics.json").write_text(json.dumps({"planned_cells": 216, "completed_cells": 216, "bootstrap_replicates": 10000, "kernel_paired_bootstrap_rows": 108, "checkpoint_degradation_rows": 180, "inference_time_total_s": float(sum(float(row["inference_time_s"]) for row in rows))}, indent=2) + "\n", encoding="utf-8")
    (campaign / "results.json").write_text(json.dumps({"status": "exploratory_held_out_complete_pending_review", "scientific_boundary": "exploratory_held_out_surrogate_only", "g2_formal_status": "not_passed", "model_promotion": False, "physical_claim": False}, indent=2) + "\n", encoding="utf-8")
    (campaign / "campaign_status.json").write_text(json.dumps({"status": "evaluation_complete_pending_review", "planned_cells": 216, "completed_cells": 216, "next_gate": "human_review_of_exploratory_surrogate_evidence"}, indent=2) + "\n", encoding="utf-8")
    return campaign


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-g1", "freeze-g2", "freeze-continuations", "freeze-g3"):
        item = sub.add_parser(name); item.add_argument("--output-dir", required=True); item.add_argument("--repository-root", default=".")
        if name in {"freeze-g1", "freeze-g2", "freeze-g3"}: item.add_argument("--config", required=True)
        if name == "freeze-g2": item.add_argument("--g1-dir", required=True)
        if name == "freeze-continuations": item.add_argument("--source-campaign", required=True)
        if name == "freeze-g3":
            item.add_argument("--g2-campaign", required=True)
            item.add_argument("--continuation-campaign", required=True)
            item.add_argument("--reuse-test-cache-from")
    for name in ("preflight-g3", "run-g3-shard", "status-g3", "finalize-g3"):
        item = sub.add_parser(name); item.add_argument("--campaign-dir", required=True); item.add_argument("--repository-root", default=".")
        if name in {"preflight-g3", "run-g3-shard"}: item.add_argument("--device", default="cuda:0")
        if name == "run-g3-shard": item.add_argument("--shard-id", required=True, type=int)
    item = sub.add_parser("run-training-task")
    item.add_argument("--manifest", required=True)
    item.add_argument("--task-index", required=True, type=int)
    item.add_argument("--repository-root", default=".")
    item.add_argument("--admission-probe", action="store_true")
    item.add_argument("--run-dir-override")
    item = sub.add_parser("audit-training-manifest")
    item.add_argument("--manifest", required=True)
    item.add_argument("--start-index", required=True, type=int)
    item.add_argument("--end-index", required=True, type=int)
    item.add_argument("--repository-root", default=".")
    args = parser.parse_args(argv)
    if args.command == "finalize-g3":
        result = finalize_g3(args.campaign_dir, repository_root=args.repository_root)
    elif args.command == "preflight-g3":
        result = preflight_g3(args.campaign_dir, repository_root=args.repository_root, device=args.device)
    elif args.command == "run-g3-shard":
        result = run_g3_shard(args.campaign_dir, args.shard_id, repository_root=args.repository_root, device=args.device)
    elif args.command == "status-g3":
        result = status_g3(args.campaign_dir, repository_root=args.repository_root)
    elif args.command == "freeze-g1":
        result = freeze_g1(args.config, args.output_dir, repository_root=args.repository_root)
    elif args.command == "freeze-g2":
        result = freeze_g2(args.config, args.g1_dir, args.output_dir, repository_root=args.repository_root)
    elif args.command == "freeze-continuations":
        result = freeze_exp_fwd_020_continuations(args.source_campaign, args.output_dir, repository_root=args.repository_root)
    elif args.command == "run-training-task":
        result = run_training_task(
            args.manifest, args.task_index, repository_root=args.repository_root,
            admission_probe=args.admission_probe, run_dir_override=args.run_dir_override,
        )
    elif args.command == "audit-training-manifest":
        result = audit_training_manifest(
            args.manifest, args.start_index, args.end_index, repository_root=args.repository_root,
        )
    else:
        result = freeze_g3(args.config, args.g2_campaign, args.continuation_campaign, args.output_dir, repository_root=args.repository_root, reuse_test_cache_from=args.reuse_test_cache_from)
    print(result)


__all__ = ["freeze_g1", "freeze_g2", "freeze_exp_fwd_020_continuations", "freeze_g3", "run_training_task", "audit_training_manifest", "preflight_g3", "run_g3_shard", "status_g3", "finalize_g3"]


if __name__ == "__main__": main()
