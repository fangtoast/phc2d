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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


VARIANTS = ("baseline", "candidate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
    expected = {(domain, variant, seed) for domain in gate.get("selected_domains", ()) for variant in VARIANTS for seed in (0, 1, 2)}
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
    if out.exists():
        raise FileExistsError(f"G3 campaign directory already exists: {out}")
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
        if result.get("status") != "formal_training_success" or result.get("test_executed"):
            raise RuntimeError(f"untouched G2 source invalid: {row['case_id']}")
        final_rows.append({**row, "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "checkpoint_origin": "g2_validation_best"})
    for row in continuation_rows:
        run = _resolve(root, row["run_dir"])
        result_path, checkpoint = run / "results.json", run / "model.pt"
        if not result_path.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"continuation result/checkpoint missing: {row['case_id']}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("test_executed") or int(result.get("stop_epoch", -1)) != 300 or int(result.get("history_epoch_count", -1)) != 300:
            raise RuntimeError(f"continuation final state invalid: {row['case_id']}")
        final_rows.append({**row, "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "checkpoint_origin": "epoch300_validation_best"})
    identities = {(row["domain"], row["variant"], int(row["seed"])) for row in final_rows}
    expected = {(row["domain"], row["variant"], int(row["seed"])) for row in g2_rows}
    if len(final_rows) != 36 or identities != expected:
        raise RuntimeError("final checkpoint inventory has a collision or omission")
    import h5py

    out.mkdir(parents=True)
    cache_dir = out / "test_cache"; cache_dir.mkdir()
    datasets: list[dict[str, Any]] = []
    for domain, contract in config["datasets"].items():
        path = _resolve(root, str(contract["path"]))
        if _sha256(path) != str(contract["sha256"]):
            raise RuntimeError(f"dataset hash drift: {domain}")
        with h5py.File(path, "r") as src:
            required = {str(contract["grid_field"]), "Wh", "idx_test", "idx_val", "k_points"}
            missing = required.difference(src.keys())
            if missing:
                raise RuntimeError(f"dataset fields missing for {domain}: {sorted(missing)}")
            indices = np.asarray(src["idx_test"], dtype=np.int64)
            if len(indices) != int(contract["test_size"]):
                raise RuntimeError(f"test size drift for {domain}")
            cache = cache_dir / f"{domain}.h5"
            with h5py.File(cache, "w") as dst:
                dst.create_dataset("input", data=src[str(contract["grid_field"])][indices], dtype="f4")
                dst.create_dataset("Wh", data=src["Wh"][indices], dtype="f4")
                dst.create_dataset("k_points", data=src["k_points"], dtype="f4")
                dst.create_dataset("test_indices", data=indices, dtype="i8")
            datasets.append({"domain": domain, "path": str(path), "grid_field": contract["grid_field"], "dataset_sha256": contract["sha256"], "idx_test_sha256": hashlib.sha256(indices.tobytes()).hexdigest(), "k_points_sha256": hashlib.sha256(np.asarray(src["k_points"], dtype=np.float32).tobytes()).hexdigest(), "cache_path": str(cache), "cache_sha256": _sha256(cache)})
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
    (out / "freeze.json").write_text(json.dumps({"experiment_id": "EXP-FWD-020", "stage": "G3", "test_cache_created_once": True, "checkpoint_count": 36, "cell_count": 216, "relationship_counts": relation_counts, "plan_sha256": _sha256(out / "evaluation_plan.csv")}, indent=2) + "\n", encoding="utf-8")
    (out / "campaign_status.json").write_text(json.dumps({"status": "frozen_preflight_required", "planned_cells": 216, "completed_cells": 0, "next_gate": "g3_six_domain_preflight"}, indent=2) + "\n", encoding="utf-8")
    return out


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
    args = parser.parse_args(argv)
    if args.command == "freeze-g1":
        result = freeze_g1(args.config, args.output_dir, repository_root=args.repository_root)
    elif args.command == "freeze-g2":
        result = freeze_g2(args.config, args.g1_dir, args.output_dir, repository_root=args.repository_root)
    elif args.command == "freeze-continuations":
        result = freeze_exp_fwd_020_continuations(args.source_campaign, args.output_dir, repository_root=args.repository_root)
    else:
        result = freeze_g3(args.config, args.g2_campaign, args.continuation_campaign, args.output_dir, repository_root=args.repository_root)
    print(result)


__all__ = ["freeze_g1", "freeze_g2", "freeze_exp_fwd_020_continuations", "freeze_g3"]


if __name__ == "__main__": main()
