"""Freeze and execute bounded, dynamically claimed single-GPU convergence tasks.

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
from phc2d.forward.full_cell_training_contracts import resolve_full_cell_training_config


def adaptive_config(config: dict, experiment_id: str, protocol: dict, source_run: Path | None = None, source_experiment: str | None = None) -> dict:
    config = copy.deepcopy(config)
    config["experiment"].update(experiment_id=experiment_id, held_out_test_authorized=False,
                                authorization_id="user-2026-10-22-start-convergence-n40-n56-eight-gpu")
    for key in ("required_stop_epoch", "convergence_window", "convergence_relative_threshold", "convergence_metric"):
        config["training"].pop(key, None)
    config["training"].update(max_epochs=1000, early_stopping_patience=0, epoch_100_continue_gate=False)
    config["adaptive_optimization"] = protocol
    config["runtime"].update(cache_in_memory=True, num_workers=0, persistent_workers=False)
    if source_run:
        config["continuation"] = dict(mode="epoch300_best_weights_optimizer_reset", source_epoch=300,
                                      source_run=str(source_run), source_experiment=source_experiment,
                                      optimizer_reset=True)
    return config


def freeze(profile_path: Path, output: Path, stamp: str) -> None:
    """Freeze source bytes, dataset identities and all paired seeds before admission."""
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    platform = profile["platform"]
    groups = []
    if platform == "n40":
        for experiment_id, parent in profile["source_campaigns"].items():
            parent = Path(parent)
            source_experiment = {"EXP-FWD-026": "EXP-FWD-024", "EXP-FWD-027": "EXP-FWD-025"}[experiment_id]
            configs = []
            for row in read_tasks(parent):
                source_run = Path(row["run_dir"])
                config = yaml.safe_load(Path(row["config"]).read_text(encoding="utf-8"))
                result = json.loads((source_run / "results.json").read_text())
                if result["stop_epoch"] != 300 or result["model"] != row["model"] or int(result["seed"]) != int(row["seed"]):
                    raise ValueError("invalid completed source identity")
                provenance = {name: file_digest(source_run / name) for name in
                              ("history.csv", "model.pt", "epoch300.pt", "results.json", "config.resolved.yaml", "run.meta.json")}
                if provenance["model.pt"] != result["checkpoint_sha256"] or provenance["epoch300.pt"] != result["terminal_checkpoint_sha256"]:
                    raise ValueError("source checkpoint hash changed")
                configs.append((row["case_id"], int(row["seed"]), row["model"],
                                adaptive_config(config, experiment_id, profile["protocol"], source_run, source_experiment), provenance))
            groups.append((experiment_id, configs))
    else:
        architecture_source = Path(profile["architecture_source"])
        architecture = yaml.safe_load(architecture_source.read_text(encoding="utf-8"))
        for name, params in profile["models"].items():
            if architecture["models"][name] != params or architecture["training"]["learning_rates"][name] != profile["learning_rates"][name]:
                raise ValueError("baseline architecture or learning rate differs from the frozen source")
        profile["architecture_source_sha256"] = file_digest(architecture_source)
        template = yaml.safe_load(Path(profile["core_template"]).read_text(encoding="utf-8"))
        configs = []
        for domain, dataset in template["datasets"].items():
            for model, params in profile["models"].items():
                for seed in (0, 1, 2):
                    config = dict(experiment=dict(experiment_id="EXP-FWD-028", lane=dataset["lane"], source_domain=domain),
                                  dataset=dict(path=dataset["path"], grid_field=dataset["grid_field"], expected_dataset_sha256=dataset["sha256"]),
                                  representation=dict(resolution=64, sampling="continuous", threshold=None, physical_equivalence=True),
                                  models={model: params}, training=copy.deepcopy(template["training"]),
                                  runtime=template["runtime"], output=dict(runs_dir="runs"))
                    config["training"]["learning_rates"] = profile["learning_rates"]
                    configs.append((f"{domain}_{model}_s{seed}", seed, model,
                                    adaptive_config(config, "EXP-FWD-028", profile["protocol"]), {}))
        groups.append(("EXP-FWD-028", configs))
    campaigns = []
    for experiment_id, configs in groups:
        expected = {"EXP-FWD-026": 54, "EXP-FWD-027": 30, "EXP-FWD-028": 108}[experiment_id]
        if len(configs) != expected:
            raise ValueError("unexpected task count")
        campaign = output / f"{experiment_id}_{stamp}"
        campaign.mkdir(parents=True, exist_ok=False)
        (campaign / "configs").mkdir()
        rows, datasets, source_records = [], {}, {}
        for case, seed, model, config, provenance in configs:
            resolve_full_cell_training_config(config, model_name=model, seed=seed, admission_probe=False)
            dataset = config["dataset"]
            if dataset["path"] not in datasets:
                datasets[dataset["path"]] = inspect_dataset(dict(path=dataset["path"], sha256=dataset["expected_dataset_sha256"], grid_field=dataset["grid_field"]),
                                                            config["training"]["train_size"], config["training"].get("train_subset_salt", "unused-full-train"))
            path = campaign / "configs" / f"{case}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            source_records[case] = provenance
            rows.append(dict(task_index=len(rows), case_id=case, model=model, seed=seed, lane=config["experiment"]["lane"],
                             config=str(path), config_sha256=file_digest(path), run_dir=str(campaign / "training" / case)))
        write_table(campaign / "task_manifest.csv", rows)
        write_json(campaign / "datasets.json", datasets)
        write_json(campaign / "source_provenance.json", source_records)
        (campaign / "config.resolved.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
        write_json(campaign / "freeze.json", dict(status="config_ready", task_count=expected,
                   manifest_sha256=file_digest(campaign / "task_manifest.csv"), profile_sha256=file_digest(profile_path),
                   test_targets_loaded=False, formal_training_started=False))
        write_json(campaign / "run.meta.json", dict(script="phc2d.forward.convergence_campaign",
                   git_commit=os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT"), experiment_id=experiment_id,
                   primary_layer="L3", platform=platform))
        campaigns.append(str(campaign))
    suite = Path(campaigns[0])
    write_json(suite / "suite.json", dict(campaigns=campaigns, platform=platform, gpu_limit=8,
               no_automatic_retry=True, test_authorized=False))
    print(json.dumps(dict(suite=str(suite), campaigns=campaigns)))


def suite_tasks(suite: Path, admission: bool = False) -> list[tuple[Path, dict]]:
    profile = json.loads((suite / "suite.json").read_text())
    groups = [read_tasks(Path(c)) for c in profile["campaigns"]]
    # Interleave campaigns to avoid starving either the core or the branch extension.
    return [(Path(c), rows[i]) for i in range(max(map(len, groups)))
            for c, rows in zip(profile["campaigns"], groups) if i < len(rows)
            and (not admission or int(rows[i]["seed"]) == 0)]


def audit_admission(suite: Path) -> None:
    for campaign, row in suite_tasks(suite, True):
        out = campaign / "admission" / row["case_id"]
        result = json.loads((out / "results.json").read_text())
        if result["status"] != "full_cell_forward_admission_probe_pass" or result["model"] != row["model"] or result["seed"] != 0:
            raise RuntimeError("admission identity or status mismatch")
        if file_digest(out / "model.pt") != result["checkpoint_sha256"] or result["test_executed"]:
            raise RuntimeError("admission checkpoint or selection boundary invalid")


def worker(suite: Path, stage: str) -> None:
    """A claim is never reused; errors preserve evidence and require an explicit retry."""
    from phc2d.forward.full_cell import run_full_cell_forward
    if stage == "train":
        audit_admission(suite)
    claim_root = suite / f"{stage}_claims"
    claim_root.mkdir(exist_ok=True)
    for campaign, row in suite_tasks(suite, stage == "admission"):
        claim = claim_root / (campaign.name + "_" + row["case_id"])
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        write_json(claim / "claim.json", dict(job_id=os.environ.get("SLURM_JOB_ID"), array_task=os.environ.get("SLURM_ARRAY_TASK_ID"), task=row))
        out = campaign / "admission" / row["case_id"] if stage == "admission" else Path(row["run_dir"])
        try:
            if out.exists():
                raise FileExistsError(out)
            config = yaml.safe_load(Path(row["config"]).read_text(encoding="utf-8"))
            source = config.get("continuation", {}).get("source_run")
            if source:
                provenance = json.loads((campaign / "source_provenance.json").read_text())[row["case_id"]]
                if any(file_digest(Path(source) / name) != digest for name, digest in provenance.items()):
                    raise RuntimeError("frozen source artifact changed after admission")
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            run_full_cell_forward(row["config"], model_name=row["model"], seed=int(row["seed"]), run_dir=out, admission_probe=stage == "admission")
            write_json(claim / "completed.json", dict(status="completed", checkpoint_sha256=file_digest(out / "model.pt")))
        except Exception as error:
            write_json(claim / "failure.json", dict(error=repr(error), retry_authorized=False))
            raise
    if stage == "train":
        from phc2d.forward.convergence_results import finish_if_complete
        finish_if_complete(suite, json.loads((suite / "suite.json").read_text())["campaigns"])


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["freeze", "admission", "train", "audit-admission"])
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stamp")
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.profile, args.output, args.stamp)
    elif args.stage == "audit-admission":
        audit_admission(args.suite)
    else:
        worker(args.suite, args.stage)


if __name__ == "__main__":
    main()
