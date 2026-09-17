"""Evaluation-only checkpoint inheritance and paired target isolation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from phc2d.forward.full_cell_multidomain_contracts import (
    audit_dataset, audit_training_runs, read_csv, repository_commit,
    sha256_array, sha256_file, write_csv,
)
from phc2d.forward.full_cell_multidomain_evaluation import (
    freeze_evaluation, load_evaluation_campaign,
)
from phc2d.runs import write_json, write_metrics, write_results, write_run_meta

TRANSFER_SCHEMA = "phc2d.full_cell_transfer_evaluation.v1"


def checkpoint_identity_map(campaign: str | Path) -> dict[str, str]:
    """Require one unique checkpoint per training identity, without path inference."""
    rows = read_csv(Path(campaign) / "evaluation/checkpoint_manifest.csv")
    identity = {r["task_id"]: r["checkpoint_sha256"] for r in rows}
    if len(identity) != len(rows) or len(set(identity.values())) != len(rows):
        raise RuntimeError("checkpoint identity/hash collision")
    return identity


def audit_transfer_cohort(
    source_campaign: str | Path, targets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify paired splits and target-test isolation from source train/validation."""
    import h5py

    source = Path(source_campaign).resolve()
    selections: dict[str, set[int]] = defaultdict(set)
    for row in read_csv(source / "subset_manifest.csv"):
        indices = np.load(source / row["indices_path"])
        if sha256_array(indices, dtype=np.int64) != row["indices_sha256"]:
            raise RuntimeError("source frozen subset hash mismatch")
        selections[row["domain"]].update(map(int, indices))
    forbidden_ids: set[str] = set()
    forbidden_grids: set[str] = set()
    source_k_hashes: set[str] = set()
    source_hashes = {}
    for row in read_csv(source / "dataset_manifest.csv"):
        path = Path(row["dataset_path"])
        digest = sha256_file(path)
        if digest != row["dataset_sha256"]:
            raise RuntimeError(f"source dataset hash mismatch: {row['domain']}")
        source_hashes[row["domain"]] = digest
        source_k_hashes.add(row["k_points_sha256"])
        with h5py.File(path, "r") as handle:
            validation = np.asarray(handle["idx_val"], dtype=np.int64)
            if sha256_array(validation, dtype=np.int64) != row["idx_val_hash"]:
                raise RuntimeError("source validation identity changed")
            indices = np.asarray(sorted(selections[row["domain"]] | set(map(int, validation))), dtype=np.int64)
            forbidden_ids.update(np.asarray(handle["sample_id"][indices]).astype("U").tolist())
            for start in range(0, len(indices), 256):
                grids = np.asarray(handle[row["grid_field"]][indices[start:start + 256]], dtype=np.float32)
                forbidden_grids.update(sha256_array(grid, dtype=np.float32) for grid in grids)
    if len(source_k_hashes) != 1:
        raise RuntimeError("source checkpoint k-grid identity is inconsistent")

    cohorts: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    counts = {}
    for domain, target in targets.items():
        lane = domain.rsplit("_", 1)[1]
        with h5py.File(target["dataset_path"], "r") as handle:
            ids = np.asarray(handle["sample_id"]).astype("U")
            splits = {key: np.asarray(handle[key], dtype=np.int64) for key in ("idx_train", "idx_val", "idx_test")}
            if sha256_array(np.asarray(handle["k_points"]), dtype=np.float32) not in source_k_hashes:
                raise RuntimeError("target/source k-grid mismatch")
            if lane in cohorts:
                old_ids, old_splits = cohorts[lane]
                if not np.array_equal(ids, old_ids) or any(not np.array_equal(splits[k], old_splits[k]) for k in splits):
                    raise RuntimeError(f"paired cohort identity/split mismatch: {domain}")
            else:
                cohorts[lane] = ids, splits
            indices = np.sort(splits["idx_test"])
            id_overlap = len(set(ids[indices].tolist()) & forbidden_ids)
            grid_overlap = 0
            for start in range(0, len(indices), 256):
                grids = np.asarray(handle[target["grid_field"]][indices[start:start + 256]], dtype=np.float32)
                grid_overlap += sum(sha256_array(grid, dtype=np.float32) in forbidden_grids for grid in grids)
            if id_overlap or grid_overlap:
                raise RuntimeError(f"target test leaks source train/validation: {domain}")
            counts[domain] = {"test_sample_count": len(indices), "sample_id_overlap": id_overlap, "input_grid_overlap": grid_overlap}
    return {
        "status": "paired_and_source_isolated", "source_dataset_sha256": source_hashes,
        "source_forbidden_identity_count": len(forbidden_ids),
        "source_forbidden_grid_count": len(forbidden_grids), "targets": counts,
        "source_test_labels_loaded": False, "target_test_labels_loaded": False,
    }


def freeze_transfer_evaluation(
    config_path: str | Path, output_dir: str | Path, *, repository_root: str | Path,
) -> Path:
    """Create a derivative campaign; only six new held-out labels enter caches."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != TRANSFER_SCHEMA or config.get("experiment_id") != "EXP-FWD-023":
        raise ValueError("unsupported transfer evaluation contract")
    if (config.get("checkpoint_count"), config.get("new_cell_count"), config.get("combined_cell_count")) != (63, 378, 756):
        raise ValueError("EXP-FWD-023 requires the frozen 63/378/756 identity contract")
    if not config.get("authorization_id"):
        raise ValueError("explicit authorization identity is required")
    source, profile = load_evaluation_campaign(config["source_campaign"])
    if profile["experiment"]["experiment_id"] != config["source_experiment_id"]:
        raise RuntimeError("source experiment identity mismatch")
    audit_training_runs(source / "task_manifest.csv")
    checkpoints = checkpoint_identity_map(source)
    if len(checkpoints) != config["checkpoint_count"]:
        raise RuntimeError("source checkpoint manifest count mismatch")
    expected = {f"k{k}_{lane}" for k in (8, 10, 15) for lane in ("binary", "graded")}
    if set(config["datasets"]) != expected or set(config["datasets"]) & set(profile["evaluation"]["targets"]):
        raise ValueError("new targets must be exactly the disjoint K8/K10/K15 six domains")
    campaign = Path(output_dir).resolve()
    if campaign == source or (campaign.exists() and any(campaign.iterdir())):
        raise FileExistsError("transfer campaign must use a new empty directory")
    pairing = Path(config["pairing_audit"])
    pairing_payload = json.loads(pairing.read_text(encoding="utf-8"))
    if pairing_payload.get("status") != "passed":
        raise RuntimeError("source pairing audit did not pass")
    audits = {}
    for domain, contract in config["datasets"].items():
        audit = audit_dataset(domain, contract, contract["path"])
        cluster_file = pairing.parent / f"{domain}_hpc_audit.json"
        cluster = json.loads(cluster_file.read_text(encoding="utf-8"))
        if cluster.get("status") != "passed" or cluster.get("problems") or cluster.get("dataset_sha256") != audit["dataset_sha256"]:
            raise RuntimeError(f"target cluster audit/hash mismatch: {domain}")
        audits[domain] = {**audit, "dataset_acceptance": config["dataset_acceptance"], "cluster_audit_sha256": sha256_file(cluster_file)}
    isolation = audit_transfer_cohort(source, audits)
    campaign.mkdir(parents=True, exist_ok=True)
    tasks = [{**r, "run_dir": str(source / r["run_dir"])} for r in read_csv(source / "task_manifest.csv")]
    normalizers = [{**r, "normalizer_path": str(source / r["normalizer_path"])} for r in read_csv(source / "normalizer_manifest.csv")]
    subsets = [{**r, "indices_path": str(source / r["indices_path"])} for r in read_csv(source / "subset_manifest.csv")]
    write_csv(campaign / "task_manifest.csv", tasks)
    write_csv(campaign / "normalizer_manifest.csv", normalizers)
    write_csv(campaign / "subset_manifest.csv", subsets)
    write_csv(campaign / "dataset_manifest.csv", [*read_csv(source / "dataset_manifest.csv"), *audits.values()])
    resolved = {**profile, "experiment": {**profile["experiment"], "experiment_id": config["experiment_id"]},
                "evaluation": {**profile["evaluation"], "targets": list(config["datasets"])},
                "transfer_evaluation": {**config, "source_checkpoint_manifest_sha256": sha256_file(source / "evaluation/checkpoint_manifest.csv"),
                                        "pairing_audit_sha256": sha256_file(pairing),
                                        "execution_start_git_commit": repository_commit(repository_root)}}
    (campaign / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(campaign / "isolation_audit.json", isolation)
    write_json(campaign / "pairing_audit.json", pairing_payload)
    write_run_meta(campaign, {"experiment_id": config["experiment_id"], "source_campaign": str(source), "execution_start_git_commit": repository_commit(repository_root)})
    (campaign / "run.log").write_text("Evaluation-only derivative; no new training or source test target reads.\n", encoding="utf-8")
    write_csv(campaign / "history.csv", [{"phase": "inherit_and_audit", "status": "complete"}])
    freeze_evaluation(campaign)
    if checkpoint_identity_map(campaign) != checkpoints:
        raise RuntimeError("derivative checkpoint identities changed")
    write_metrics(campaign, {"checkpoint_count": len(checkpoints), "new_cell_count": 378, "combined_cell_count": 756, "dataset_acceptance": config["dataset_acceptance"]})
    write_results(campaign, {"status": "extension_frozen_pending_cells", "model_promotion": False, "physical_claim": False})
    return campaign
