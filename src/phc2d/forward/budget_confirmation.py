"""Freeze and execute independent fixed-budget confirmation tasks.

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
from typing import Any

import numpy as np
import yaml

from phc2d.data.sampling import select_nested_sample_indices
from phc2d.forward.full_cell_training_contracts import resolve_full_cell_training_config


def file_digest(path: Path) -> str:
    """Hash an immutable input without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_dataset(spec: dict[str, Any], train_size: int, salt: str) -> dict[str, Any]:
    """Verify bytes, split disjointness and train-only normalization inputs."""
    import h5py

    path = Path(spec["path"])
    digest = file_digest(path)
    if digest != spec["sha256"]:
        raise ValueError(f"dataset SHA-256 mismatch: {path}")
    with h5py.File(path, "r") as handle:
        splits = [np.asarray(handle[key], dtype=np.int64) for key in ("idx_train", "idx_val", "idx_test")]
        n = len(handle["Wh"])
        if any(len(np.unique(s)) != len(s) or np.any(s < 0) or np.any(s >= n) for s in splits):
            raise ValueError("invalid split indices")
        if any(np.intersect1d(splits[i], splits[j]).size for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("train/validation/test overlap")
        indices = np.sort(splits[0])
        if train_size > len(indices):
            raise ValueError("training size exceeds frozen training split")
        if train_size < len(indices):
            ids = np.asarray(handle["sample_id"][indices]).astype("U").tolist()
            indices = select_nested_sample_indices(indices, ids, size=train_size, salt=salt)
        # Only training labels are read here. Test indices and shapes are metadata.
        targets = np.asarray(handle["Wh"][indices], dtype=np.float32)
        mean = np.mean(targets, axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(np.std(targets, axis=0, dtype=np.float64).astype(np.float32), np.float32(1e-6))
        if not np.isfinite(targets).all():
            raise ValueError("non-finite training labels")
        return {"path": str(path), "sha256": digest, "grid_field": spec["grid_field"],
                "grid_shape": list(handle[spec["grid_field"]].shape[1:]),
                "target_shape": list(handle["Wh"].shape[1:]), "train_count": len(indices),
                "validation_count": len(splits[1]), "test_count": len(splits[2]),
                "train_subset_hash": hashlib.sha256(indices.tobytes()).hexdigest(),
                "normalizer_hash": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest(),
                "k_points_hash": hashlib.sha256(np.asarray(handle["k_points"], dtype=np.float32).tobytes()).hexdigest(),
                "test_targets_loaded": False}


def freeze_campaign(profile_path: Path, campaign: Path) -> None:
    """Create a new campaign; historical configs and runs remain immutable."""
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if "base_profile" in profile:
        override = profile
        base = profile_path.parent / override["base_profile"]
        profile = yaml.safe_load(base.read_text(encoding="utf-8"))
        if "base_profile" in profile:
            raise ValueError("nested execution profile inheritance is unsupported")
        profile["base_profile_sha256"] = file_digest(base)
        profile["platform"] = override["platform"]
        profile["execution_platform_basis"] = override["execution_platform_basis"]
        for spec in profile["datasets"].values():
            spec["path"] = str(Path(override["dataset_path_root"]) / Path(spec["path"]).name)
    if campaign.exists():
        raise FileExistsError(campaign)
    if not profile.get("authorization_id") or profile["seeds"] != [0, 1, 2]:
        raise ValueError("authorization and paired three-seed contract required")
    experiment_id = profile["experiment_id"]
    configs: list[tuple[str, int, dict[str, Any]]] = []
    dataset_records: dict[str, Any] = {}
    if experiment_id == "EXP-FWD-024":
        if len(profile["datasets"]) != 6 or len(profile["models"]) != 3:
            raise ValueError("confirmation requires six domains and three core models")
        for domain, spec in profile["datasets"].items():
            record = inspect_dataset(spec, profile["training"]["train_size"], profile["training"]["train_subset_salt"])
            dataset_records[domain] = record
            for model in profile["models"]:
                for seed in profile["seeds"]:
                    config = {"experiment": {"experiment_id": experiment_id, "lane": spec["lane"], "source_domain": domain},
                              "dataset": {"path": spec["path"], "grid_field": spec["grid_field"], "expected_dataset_sha256": spec["sha256"]},
                              "representation": {"resolution": record["grid_shape"][0], "sampling": "continuous", "threshold": None, "physical_equivalence": True},
                              "models": profile["models"], "training": profile["training"], "runtime": profile["runtime"], "output": {"runs_dir": "runs"}}
                    configs.append((f"{domain}_{model}_s{seed}", seed, config))
    elif experiment_id == "EXP-FWD-025":
        parent = Path(profile["source_campaign"])
        rows = list(csv.DictReader((parent / "task_manifest.csv").open(encoding="utf-8")))
        if len(rows) != 30:
            raise ValueError("branch confirmation requires all 30 parent tasks")
        for row in rows:
            source_path = parent / "derived_configs" / f"{row['case_id']}.yaml"
            if file_digest(source_path) != row["config_sha256"]:
                raise ValueError("parent config changed")
            config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            spec = {"path": config["dataset"]["path"], "sha256": config["dataset"]["expected_dataset_sha256"], "grid_field": config["dataset"]["grid_field"]}
            lane = config["experiment"]["lane"]
            if lane not in dataset_records:
                dataset_records[lane] = inspect_dataset(spec, config["training"]["train_size"], "unused-full-train")
            config["experiment"].update(experiment_id=experiment_id, parent_experiment="EXP-FWD-022", held_out_test_authorized=False)
            config["training"].update(profile["training"])
            config["runtime"].update(profile["runtime"])
            configs.append((row["case_id"], int(row["seed"]), config))
    else:
        raise ValueError("unsupported confirmation experiment")
    campaign.mkdir(parents=True)
    (campaign / "configs").mkdir()
    tasks = []
    for case, seed, config in configs:
        model = next(iter(config["models"])) if experiment_id == "EXP-FWD-025" else case.rsplit("_s", 1)[0].split(config["experiment"]["source_domain"] + "_", 1)[1]
        config["models"] = {model: config["models"][model]}
        config["experiment"]["authorization_id"] = profile["authorization_id"]
        resolve_full_cell_training_config(config, model_name=model, seed=seed, admission_probe=False)
        path = campaign / "configs" / f"{case}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        tasks.append({"task_index": len(tasks), "case_id": case, "model": model, "seed": seed,
                      "lane": config["experiment"]["lane"], "config": str(path), "config_sha256": file_digest(path),
                      "run_dir": str(campaign / "training" / case)})
    write_table(campaign / "task_manifest.csv", tasks)
    write_json(campaign / "datasets.json", dataset_records)
    (campaign / "config.resolved.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    write_json(campaign / "freeze.json", {"status": "config_ready", "task_count": len(tasks), "authorization_id": profile["authorization_id"],
              "profile_sha256": file_digest(profile_path), "manifest_sha256": file_digest(campaign / "task_manifest.csv"),
              "test_targets_loaded": False, "formal_training_started": False})


def read_tasks(campaign: Path) -> list[dict[str, str]]:
    frozen = json.loads((campaign / "freeze.json").read_text(encoding="utf-8"))
    if file_digest(campaign / "task_manifest.csv") != frozen["manifest_sha256"]:
        raise ValueError("frozen manifest changed")
    rows = list(csv.DictReader((campaign / "task_manifest.csv").open(encoding="utf-8")))
    for row in rows:
        if file_digest(Path(row["config"])) != row["config_sha256"]:
            raise ValueError("frozen task config changed")
    return rows


def audit_campaign(campaign: Path, admission: bool = False) -> dict[str, Any]:
    """Audit every expected artifact before permitting the next stage."""
    from phc2d.runs import validate_run_artifacts
    rows = read_tasks(campaign)
    if admission:
        rows = [row for row in rows if int(row["seed"]) == 0]
    records = []
    for row in rows:
        path = campaign / "admission" / row["case_id"] if admission else Path(row["run_dir"])
        if validate_run_artifacts(path):
            raise RuntimeError(f"missing artifacts: {path}")
        result = json.loads((path / "results.json").read_text(encoding="utf-8"))
        history = list(csv.DictReader((path / "history.csv").open(encoding="utf-8")))
        expected_epochs = 1 if admission else 300
        if [int(r["epoch"]) for r in history] != list(range(1, expected_epochs + 1)):
            raise RuntimeError(f"incomplete epochs: {path}")
        if (result.get("test_executed") or result.get("validation_overlap")
                or int(result["seed"]) != int(row["seed"]) or result["model"] != row["model"]):
            raise RuntimeError("selection boundary or task identity mismatch")
        for record in history:
            if not np.isfinite([float(record[k]) for k in ("train_loss", "val_bands_2_10_mre_pct")]).all():
                raise RuntimeError("non-finite training history")
        if not admission and not (path / "epoch300.pt").is_file():
            raise RuntimeError("terminal checkpoint missing")
        records.append({"case_id": row["case_id"], "checkpoint_sha256": file_digest(path / "model.pt"), "status": result["status"]})
    report = {"status": "admission_pass" if admission else "training_complete_fixed_budget", "task_count": len(records),
              "test_targets_loaded": False, "convergence_is_not_implied": True, "records": records}
    write_json(campaign / ("admission_audit.json" if admission else "training_audit.json"), report)
    return report


def run_bundle(campaign: Path, slot: int, slots: int, admission: bool = False) -> None:
    """Run disjoint single-GPU tasks sequentially within a scheduler slot."""
    from phc2d.forward.full_cell import run_full_cell_forward
    if slots <= 0 or not 0 <= slot < slots:
        raise ValueError("invalid bundle slot")
    rows = read_tasks(campaign)
    if admission:
        rows = [row for row in rows if int(row["seed"]) == 0]
    else:
        audit_campaign(campaign, admission=True)
    for row in rows[slot::slots]:
        output = campaign / "admission" / row["case_id"] if admission else Path(row["run_dir"])
        if output.exists():
            raise FileExistsError(f"refusing to retry or overwrite {output}")
        run_full_cell_forward(row["config"], model_name=row["model"], seed=int(row["seed"]), run_dir=output, admission_probe=admission)
    if not admission:
        finish_native_when_complete(campaign, slot, slots)


def freeze_native_test(campaign: Path) -> None:
    """Freeze all native test checkpoints after the complete training audit."""
    profile = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    if profile["experiment_id"] != "EXP-FWD-024":
        raise ValueError("test evaluation is not authorized for branch confirmation")
    audit_campaign(campaign)
    path = campaign / "test_manifest.csv"
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for task in read_tasks(campaign):
        checkpoint = Path(task["run_dir"]) / "model.pt"
        rows.append({"task_index": len(rows), "task_id": task["case_id"], "model": task["model"],
                     "seed": task["seed"], "lane": "B" if task["lane"] == "binary" else "G",
                     "head": "dense", "source_kind": "EXP-FWD-024_fixed300_native_8k",
                     "source_run_dir": task["run_dir"], "checkpoint_path": str(checkpoint),
                     "checkpoint_sha256": file_digest(checkpoint), "result_dir": str(campaign / "test" / task["case_id"])})
    write_table(path, rows)
    write_json(campaign / "test_freeze.json", {"manifest_sha256": file_digest(path), "checkpoint_count": len(rows),
               "test_targets_loaded": False, "historical_test_previously_viewed": True})


def run_test_bundle(campaign: Path, slot: int, slots: int) -> None:
    """Reuse the native Full-Wh evaluator with immutable checkpoint identities."""
    from phc2d.forward.full_cell_upgrade_test import run_test_task
    if slots <= 0 or not 0 <= slot < slots:
        raise ValueError("invalid bundle slot")
    path = campaign / "test_manifest.csv"
    frozen = json.loads((campaign / "test_freeze.json").read_text(encoding="utf-8"))
    if file_digest(path) != frozen["manifest_sha256"]:
        raise ValueError("test manifest changed")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows[slot::slots]:
        run_test_task(path, task_index=int(row["task_index"]))


def summarize_native_test(campaign: Path) -> None:
    """Write bounded seed-level native results after all expected tests finish."""
    manifest = list(csv.DictReader((campaign / "test_manifest.csv").open(encoding="utf-8")))
    cells = []
    for row in manifest:
        path = Path(row["result_dir"])
        result = json.loads((path / "results.json").read_text(encoding="utf-8"))
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if not result["test_executed"] or result["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("test evidence identity mismatch")
        values = metrics["evaluations"][0]
        per_sample = list(csv.DictReader((path / "per_sample_metrics.csv").open(encoding="utf-8")))
        per_band = list(csv.DictReader((path / "per_band_metrics.csv").open(encoding="utf-8")))
        config = yaml.safe_load((Path(row["source_run_dir"]) / "config.resolved.yaml").read_text(encoding="utf-8"))
        cells.append({"source_domain": config["experiment"]["source_domain"], "model": row["model"], "seed": int(row["seed"]),
                      "band1_mre_pct": float(per_band[0]["mre_pct"]),
                      "bands_2_10_mre_pct": values["test_bands_2_10_mre_pct"],
                      "bands_1_10_mre_pct": values["test_full_matrix_mre_pct"],
                      "p95_sample_bands_2_10_mre_pct": float(np.percentile([float(r["bands_2_10_mre_pct"]) for r in per_sample], 95)),
                      "mae": values["test_mae"], "checkpoint_sha256": row["checkpoint_sha256"]})
    write_table(campaign / "native_test_cells.csv", cells)
    summaries = []
    for domain, model in sorted({(r["source_domain"], r["model"]) for r in cells}):
        group = [r for r in cells if r["source_domain"] == domain and r["model"] == model]
        if {r["seed"] for r in group} != {0, 1, 2} or len(group) != 3:
            raise ValueError("native summary requires exactly three distinct seeds")
        item: dict[str, Any] = {"source_domain": domain, "model": model, "seed_count": 3}
        for metric in ("band1_mre_pct", "bands_2_10_mre_pct", "bands_1_10_mre_pct", "p95_sample_bands_2_10_mre_pct", "mae"):
            item[metric + "_mean"] = float(np.mean([r[metric] for r in group]))
            item[metric + "_seed_sd"] = float(np.std([r[metric] for r in group], ddof=1))
        summaries.append(item)
    write_table(campaign / "native_test_summary.csv", summaries)
    write_json(campaign / "results.json", {"status": "fixed300_native_surrogate_evaluation_complete", "training_task_count": len(cells),
               "test_cell_count": len(cells), "seed_count": 3, "historical_test_previously_viewed": True,
               "convergence_is_not_implied": True, "physical_claim": False, "model_promotion": False})
    from phc2d.forward.native_test_reporting import export_primary_reports
    export_primary_reports(campaign)


def finish_native_when_complete(campaign: Path, slot: int, slots: int) -> None:
    """Let the last successful bundle freeze and test all models once, without retries."""
    profile = yaml.safe_load((campaign / "config.resolved.yaml").read_text(encoding="utf-8"))
    if profile["experiment_id"] != "EXP-FWD-024":
        return
    identity = json.loads((campaign / "freeze.json").read_text(encoding="utf-8"))["manifest_sha256"]
    marker = campaign / f"bundle_completed_{slot}.json"
    temporary = marker.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump({"slot": slot, "slots": slots, "manifest_sha256": identity}, handle)
    try:
        os.link(temporary, marker)
    finally:
        temporary.unlink()
    for index in range(slots):
        marker = campaign / f"bundle_completed_{index}.json"
        if not marker.is_file():
            return
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record != {"slot": index, "slots": slots, "manifest_sha256": identity}:
            raise ValueError("bundle completion identity mismatch")
    try:
        claim = (campaign / "native_test_finalizer_claim.json").open("x", encoding="utf-8")
    except FileExistsError:
        return
    with claim:
        json.dump({"slot": slot, "status": "finalization_intent_no_automatic_retry"}, claim)
    freeze_native_test(campaign)
    run_test_bundle(campaign, 0, 1)
    summarize_native_test(campaign)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "admission", "train", "audit-admission", "audit-training", "freeze-test", "test"])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.profile is None:
            parser.error("freeze requires --profile")
        freeze_campaign(args.profile, args.campaign)
    elif args.stage == "freeze-test":
        freeze_native_test(args.campaign)
    elif args.stage == "test":
        run_test_bundle(args.campaign, args.slot, args.slots)
    elif args.stage.startswith("audit"):
        report = audit_campaign(args.campaign, admission=args.stage == "audit-admission")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}))
    else:
        run_bundle(args.campaign, args.slot, args.slots, admission=args.stage == "admission")


if __name__ == "__main__":
    main()
