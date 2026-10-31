"""Validate and test frozen EXP-FWD-032 checkpoints on six target domains.

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
import statistics
from pathlib import Path

from phc2d.forward.budget_confirmation import file_digest, read_tasks, write_json, write_table
from phc2d.forward.official_comparison_campaign import claim_task
from phc2d.forward.official_comparison_test import evaluate, ready_checkpoint, test_indices
from phc2d.forward.official_comparison_test_audit import audit_cell
from phc2d.forward.official_comparison_validation import report_per_band_validation


def validation_worker(suite: Path) -> None:
    """Re-evaluate selected validation checkpoints with exclusive claims."""
    rows = read_tasks(suite)
    audit = json.loads((suite / "training_audit.json").read_text())
    if audit["task_count"] != 36 or any(not (suite / "train_claims" / (suite.name + "_" + row["case_id"]) / "completed.json").is_file() for row in rows):
        raise ValueError("all training must be complete before validation export")
    for row in rows:
        out = Path(row["run_dir"])
        marker = suite / "train_claims" / (suite.name + "_" + row["case_id"]) / "completed.json"
        claim = claim_task(suite / "validation_claims", row["case_id"], dict(job_id=os.environ.get("SLURM_JOB_ID")))
        if claim is None:
            continue
        try:
            if (out / "per_band_validation.json").exists():
                raise FileExistsError(out / "per_band_validation.json")
            result = report_per_band_validation(out, batch_size=16, device="cuda:0", expected_experiment="EXP-FWD-032")
            if result["checkpoint_sha256"] != json.loads(marker.read_text())["checkpoint_sha256"]:
                raise ValueError("training marker and validation checkpoint differ")
            write_json(claim / "completed.json", dict(status="completed", checkpoint_sha256=result["checkpoint_sha256"]))
        except Exception as exc:
            write_json(claim / "failure.json", dict(error=repr(exc), retry_authorized=False))
            raise


def freeze_test(suite: Path, authorization_id: str) -> None:
    """Bind completed checkpoints to verified target bytes and frozen test indices."""
    import h5py
    import numpy as np

    if not authorization_id:
        raise ValueError("explicit test authorization is required")
    audit = json.loads((suite / "training_audit.json").read_text())
    if audit["task_count"] != 36 or audit["status"] != "training_artifacts_complete_pending_independent_review":
        raise ValueError("all 36 local training artifacts must be audited")
    tasks = {r["case_id"]: r for r in read_tasks(suite)}
    freeze = json.loads((suite / "freeze.json").read_text())
    target_path = suite / "test_target_manifest.csv"
    if file_digest(target_path) != freeze["test_target_manifest_sha256"]:
        raise ValueError("planned test target manifest changed")
    with target_path.open(newline="", encoding="utf-8") as stream:
        targets = list(csv.DictReader(stream))
    if len(targets) != 216 or len({(r["case_id"], r["target"]) for r in targets}) != 216:
        raise ValueError("test target identities incomplete")
    source_data = json.loads((suite / "datasets.json").read_text())
    verified_targets, rows = {}, []
    for entry in targets:
        row = tasks[entry["case_id"]]
        marker = suite / "train_claims" / (suite.name + "_" + row["case_id"]) / "completed.json"
        result = ready_checkpoint(Path(row["run_dir"]), marker)
        key = entry["target"]
        if key not in verified_targets:
            path = Path(entry["dataset"])
            if file_digest(path) != entry["dataset_sha256"]:
                raise ValueError(f"target data hash mismatch: {path}")
            with h5py.File(path, "r") as handle:
                external = entry["external"] == "True"
                indices = test_indices(handle, external=external)
                k_hash = hashlib.sha256(np.asarray(handle["k_points"], dtype=np.float32).tobytes()).hexdigest()
                verified_targets[key] = dict(dataset=entry["dataset"], grid_field=entry["grid_field"], external=entry["external"],
                    dataset_sha256=entry["dataset_sha256"],
                    target_shape=list(handle["Wh"].shape[1:]), grid_shape=list(handle[entry["grid_field"]].shape[1:]),
                    sample_count=len(indices), split_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), k_points_hash=k_hash)
        target = verified_targets[key]
        if (any(entry[k] != target[k] for k in ("dataset", "grid_field", "external", "dataset_sha256"))
                or target["target_shape"] != [81, 10]
                or target["grid_shape"] != [64, 64] or target["k_points_hash"] != source_data[row["domain"]]["k_points_hash"]):
            raise ValueError("target shape, hash, or k-point identity changed")
        rows.append(dict(**{k: entry[k] for k in ("case_id", "target", "dataset", "dataset_sha256", "grid_field", "external")},
                         source_run=row["run_dir"], model=row["model"], seed=row["seed"], domain=row["domain"], variant=row["variant"],
                         checkpoint_sha256=result["checkpoint_sha256"], sample_count=target["sample_count"],
                         split_sha256=target["split_sha256"], result_dir=str(suite / "test" / row["case_id"] / key)))
    path = suite / "test_manifest.csv"
    if path.exists() or (suite / "test_freeze.json").exists():
        raise FileExistsError(path)
    write_table(path, rows)
    write_json(suite / "test_freeze.json", dict(status="test_config_ready", authorization_id=authorization_id,
               cell_count=len(rows), manifest_sha256=file_digest(path), target_plan_sha256=file_digest(target_path),
               targets=verified_targets, test_labels_loaded=False, training_config_unchanged=True))


def test_rows(suite: Path) -> list[dict]:
    freeze = json.loads((suite / "test_freeze.json").read_text())
    path = suite / "test_manifest.csv"
    if not freeze.get("authorization_id") or file_digest(path) != freeze["manifest_sha256"]:
        raise ValueError("test authorization or frozen manifest changed")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 216 or len({(r["case_id"], r["target"]) for r in rows}) != 216:
        raise ValueError("test matrix is incomplete")
    return rows


def test_worker(suite: Path) -> None:
    for row in test_rows(suite):
        claim = claim_task(suite / "test_claims", row["case_id"] + "_" + row["target"],
                           dict(job_id=os.environ.get("SLURM_JOB_ID")))
        if claim is None:
            continue
        try:
            run = Path(row["source_run"])
            if file_digest(run / "model.pt") != row["checkpoint_sha256"]:
                raise ValueError("frozen checkpoint changed")
            marker = suite / "train_claims" / (suite.name + "_" + row["case_id"]) / "completed.json"
            record = evaluate(run, marker, Path(row["dataset"]), row["dataset_sha256"], row["grid_field"],
                              Path(row["result_dir"]), external=row["external"] == "True", save_predictions=True,
                              expected_experiment="EXP-FWD-032")
            if record["target_split_sha256"] != row["split_sha256"] or record["sample_count"] != int(row["sample_count"]):
                raise ValueError("test output differs from frozen target split")
            write_json(claim / "completed.json", dict(status="completed", checkpoint_sha256=row["checkpoint_sha256"]))
        except Exception as exc:
            write_json(claim / "failure.json", dict(error=repr(exc), retry_authorized=False))
            raise


def audit_test(suite: Path) -> dict:
    groups, cells = {}, []
    for row in test_rows(suite):
        claim = suite / "test_claims" / (row["case_id"] + "_" + row["target"])
        marker = json.loads((claim / "completed.json").read_text())
        if (claim / "failure.json").exists() or marker["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("test completion claim is invalid")
        label = {"exp_dat_007_12k": "dat007_12000", "exp_dat_008_24k": "dat008_24000"}.get(row["target"], row["target"])
        source = dict(run_dir=row["source_run"], model=row["model"], case_id=row["case_id"], seed=row["seed"])
        output = Path(row["result_dir"])
        cell = audit_cell(output, source, label, row["dataset_sha256"], int(row["sample_count"]))
        if cell["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError("test checkpoint differs from manifest")
        export = json.loads((output / "prediction_export.json").read_text())
        if (export["checkpoint_sha256"] != row["checkpoint_sha256"] or export["data_sha256"] != row["dataset_sha256"]
                or export["idx_hash"] != row["split_sha256"] or export["sample_count"] != int(row["sample_count"])
                or set(export["files_sha256"]) != {"predictions.npy", "targets.npy", "sample_indices.npy"}
                or any(file_digest(output / name) != sha for name, sha in export["files_sha256"].items())):
            raise ValueError("prediction export provenance differs from manifest")
        cells.append(cell)
        groups.setdefault((row["domain"], row["variant"], row["target"]), []).append(cell)
    summaries = []
    for (domain, variant, target), group in sorted(groups.items()):
        if len(group) != 3 or {c["seed"] for c in group} != {0, 1, 2}:
            raise ValueError("incomplete three-seed group")
        values = [c["bands_1_10_mre_pct"] for c in group]
        summaries.append(dict(domain=domain, variant=variant, target=target, mean=statistics.fmean(values),
                              sample_sd=statistics.stdev(values)))
    if len(summaries) != 72:
        raise ValueError("expected 72 three-seed groups")
    record = dict(status="complete_pending_independent_review", cell_count=len(cells),
                  three_seed_group_count=len(summaries), cells=cells, summaries=summaries,
                  surrogate_only=True, physical_validation=False)
    write_json(suite / "test_audit.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["validation", "freeze-test", "test", "audit-test"])
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--authorization-id")
    args = parser.parse_args()
    if args.stage == "validation":
        validation_worker(args.suite)
    elif args.stage == "freeze-test":
        freeze_test(args.suite, args.authorization_id)
    elif args.stage == "test":
        test_worker(args.suite)
    else:
        print(json.dumps({k: v for k, v in audit_test(args.suite).items() if k not in {"cells", "summaries"}}))


if __name__ == "__main__":
    main()
