"""Freeze and execute the four-domain official/operator comparison campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import yaml
from phc2d.forward.budget_confirmation import file_digest, inspect_dataset, read_tasks, write_json, write_table
from phc2d.forward.convergence_campaign import audit_admission
from phc2d.forward.full_cell_training_contracts import resolve_full_cell_training_config

MODELS = ("official_fno_band_v1", "official_tucker_tfno_band_v1", "global_local_channel_gate_v1")
PARTITIONS = {"n40": ("k5_binary", "f50_graded"), "n56": ("k5_graded", "f50_binary")}
AUTHORIZATION = "user-2026-10-27-implement-exp-fwd-031-two-servers-sixteen-gpu"


def load_profile(path: Path) -> dict:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = yaml.safe_load((path.parent / profile.pop("base_profile")).read_text(encoding="utf-8"))
    base.update(profile)
    return base


def task_configs(profile: dict) -> list[tuple[str, str, int, dict]]:
    if profile["experiment_id"] != "EXP-FWD-031" or profile["authorization_id"] != AUTHORIZATION:
        raise ValueError("unregistered experiment or authorization")
    if tuple(profile["domains"]) != PARTITIONS[profile["platform"]] or tuple(profile["models"]) != MODELS:
        raise ValueError("frozen model/domain partition changed")
    if profile["seeds"] != [0, 1, 2] or profile["test_authorized"] or profile["gpu_limit"] != 8:
        raise ValueError("invalid seed, test or GPU contract")
    training = profile["training"]
    if training["loss"] != "normalized_full_matrix_mse" or training["checkpoint_selection"] != "validation_bands_2_10_mre":
        raise ValueError("loss and selection must preserve the historical contract")
    if training["scheduler_metric"] != "val_bands_2_10_mre_pct":
        raise ValueError("scheduler must monitor bands 2-10")
    tasks = []
    for domain in profile["domains"]:
        spec = profile["datasets"][domain]
        for name in MODELS:
            for seed in profile["seeds"]:
                path = str(Path(profile["dataset_root"]) / Path(spec["path"]).name)
                config = dict(
                    experiment=dict(experiment_id="EXP-FWD-031", authorization_id=AUTHORIZATION,
                                    lane=spec["lane"], source_domain=domain, held_out_test_authorized=False,
                                    primary_reporting_metric="bands_1_10_mre_pct"),
                    dataset=dict(path=path, grid_field=spec["grid_field"], expected_dataset_sha256=spec["sha256"]),
                    representation=dict(resolution=64, sampling="continuous", threshold=None, physical_equivalence=True),
                    models={name: copy.deepcopy(profile["models"][name])}, training=copy.deepcopy(training),
                    runtime=copy.deepcopy(profile["runtime"]), adaptive_optimization=copy.deepcopy(profile["protocol"]),
                    output=dict(runs_dir="runs"))
                resolve_full_cell_training_config(config, model_name=name, seed=seed, admission_probe=False)
                tasks.append((f"{domain}_{name}_s{seed}", name, seed, config))
    return tasks


def freeze(profile_path: Path, output: Path) -> None:
    profile = load_profile(profile_path)
    tasks = task_configs(profile)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "configs").mkdir()
    datasets, rows = {}, []
    for case, name, seed, config in tasks:
        domain = config["experiment"]["source_domain"]
        if domain not in datasets:
            spec = config["dataset"]
            datasets[domain] = inspect_dataset(dict(path=spec["path"], grid_field=spec["grid_field"],
                                                  sha256=spec["expected_dataset_sha256"]),
                                              config["training"]["train_size"], config["training"]["train_subset_salt"])
            expected = 1000 if domain.startswith("k5_") else 5000
            if datasets[domain]["validation_count"] != expected or datasets[domain]["target_shape"] != [81, 10]:
                raise ValueError("dataset validation count or target changed")
        path = output / "configs" / f"{case}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append(dict(task_index=len(rows), case_id=case, model=name, seed=seed,
                         lane=config["experiment"]["lane"], config=str(path), config_sha256=file_digest(path),
                         run_dir=str(output / "training" / case)))
    write_table(output / "task_manifest.csv", rows)
    write_json(output / "datasets.json", datasets)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    write_json(output / "suite.json", dict(campaigns=[str(output)], platform=profile["platform"], gpu_limit=8,
                                         no_automatic_retry=True, test_authorized=False))
    write_json(output / "freeze.json", dict(status="config_ready", task_count=len(rows), test_targets_loaded=False,
                                          manifest_sha256=file_digest(output / "task_manifest.csv")))


def claim_task(directory: Path, name: str, record: dict) -> Path | None:
    claim = directory / name
    try:
        claim.mkdir(parents=True)
    except FileExistsError:
        return None
    write_json(claim / "claim.json", record)
    return claim


def audit_comparison_admission(suite: Path) -> None:
    audit_admission(suite)
    rows = [row for row in read_tasks(suite) if int(row["seed"]) == 0]
    if len(rows) != 6:
        raise ValueError("expected six local admission tasks")
    for row in rows:
        out = suite / "admission" / row["case_id"]
        record = json.loads((out / "contract_audit.json").read_text())
        if not record["passed"] or record["config_sha256"] != row["config_sha256"]:
            raise ValueError("admission contract identity changed")
    write_json(suite / "admission_audit.json", dict(passed=True, task_count=6, validation_only=True,
                                                  manifest_sha256=file_digest(suite / "task_manifest.csv")))


def audit_global_release(suite: Path) -> None:
    release = json.loads((suite / "global_admission_release.json").read_text())
    if release["authorization_id"] != AUTHORIZATION or release["passed_tasks"] != 12 or not release["formal_training_authorized"]:
        raise ValueError("both-platform admission release required")
    platform = json.loads((suite / "suite.json").read_text())["platform"]
    expected = release["platforms"][platform]
    if expected["manifest_sha256"] != file_digest(suite / "task_manifest.csv") or expected["admission_sha256"] != file_digest(suite / "admission_audit.json"):
        raise ValueError("global release does not bind the current local admission")


def audit_run(out: Path, row: dict, admission: bool) -> None:
    import torch
    from phc2d.models import build_model
    from phc2d.forward.training_diagnostics import gate_record, parameter_record
    cfg = yaml.safe_load((out / "config.resolved.yaml").read_text())
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    model = build_model(row["model"], input_shape=payload["input_shape"], output_shape=payload["output_shape"], **cfg["model"]["params"])
    model.load_state_dict(payload["state_dict"], strict=True)
    gates = gate_record(model)
    updated = True if not gates else all(bool(torch.count_nonzero(block.fusion.channel_logits.detach())) for block in model.blocks)
    if admission and not updated:
        raise ValueError("channel gates did not update during admission")
    write_json(out / "contract_audit.json", dict(passed=True, config_sha256=row["config_sha256"],
        selection_metric="validation_bands_2_10_mre", loss="normalized_full_matrix_mse",
        gate_update_passed=updated, gates=gates, **parameter_record(model)))
    if not admission:
        required = ["epoch300.pt", "phase300_best.pt", "phase300_evidence.json", "training_state.pt"]
        if any(not (out / name).is_file() for name in required):
            raise ValueError("missing phase or full-state evidence")


def worker(suite: Path, stage: str) -> None:
    from phc2d.forward.full_cell import run_full_cell_forward
    if stage == "train":
        local_audit = json.loads((suite / "admission_audit.json").read_text())
        if not local_audit["passed"] or local_audit["task_count"] != 6:
            raise ValueError("six local admission tasks must pass before formal training")
        audit_global_release(suite)
    for row in read_tasks(suite):
        if stage == "admission" and int(row["seed"]) != 0:
            continue
        claim = claim_task(suite / f"{stage}_claims", suite.name + "_" + row["case_id"],
                           dict(job_id=os.environ.get("SLURM_JOB_ID"), task=row))
        if claim is None:
            continue
        out = suite / "admission" / row["case_id"] if stage == "admission" else Path(row["run_dir"])
        try:
            if out.exists() or file_digest(Path(row["config"])) != row["config_sha256"]:
                raise ValueError("output exists or frozen config changed")
            run_full_cell_forward(row["config"], model_name=row["model"], seed=int(row["seed"]),
                                  run_dir=out, admission_probe=stage == "admission")
            audit_run(out, row, stage == "admission")
            write_json(claim / "completed.json", dict(status="completed", checkpoint_sha256=file_digest(out / "model.pt")))
        except Exception as exc:
            write_json(claim / "failure.json", dict(error=repr(exc), retry_authorized=False))
            raise
    if stage == "train":
        from phc2d.forward.convergence_results import finish_if_complete
        finish_if_complete(suite, [str(suite)])


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["freeze", "admission", "audit-admission", "train"])
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.profile, args.suite)
    elif args.stage == "audit-admission":
        audit_comparison_admission(args.suite)
    else:
        worker(args.suite, args.stage)


if __name__ == "__main__":
    main()
