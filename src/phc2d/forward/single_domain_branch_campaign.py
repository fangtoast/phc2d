"""Freeze and execute EXP-FWD-032 single-domain branch training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from phc2d.forward.budget_confirmation import file_digest, inspect_dataset, read_tasks, write_json, write_table
from phc2d.forward.full_cell_training_contracts import resolve_full_cell_training_config
from phc2d.forward.official_comparison_campaign import audit_run, claim_task
from phc2d.forward.single_domain_branch_matrix import build_matrix

AUTHORIZATION = "user-2026-10-31-exp-fwd-032-admission-training"
PLATFORM_RELEASE_AUTHORIZATION = "user-2026-10-31-exp-fwd-032-independent-platform-release"
ROOTS = {"n40": "/HOME/scw6g7a/run/phc2d/data", "n56": "/data/run01/scxl892/data/exp_fwd_020"}
EXTERNAL_TARGETS = {
    "exp_dat_007_12k": ("exp_dat_007_single_hole_six_family_bz9_12000_bingxing_v1.h5", "badd2081180406d6eaac451d758c8d17259a23e13a2c00198b2c82059c9b85a8"),
    "exp_dat_008_24k": ("exp_dat_008_pcsel_twelve_family_bz9_24000_bingxing_v1.h5", "8c3721e03603a01dd5a8e5c8f6b183e8c70259a259e18ac2f53483585e04aa5c"),
}


def task_configs(profile: dict, source: dict, adaptive: dict, platform: str) -> list[tuple[dict, dict]]:
    """Resolve the 36 independent training configurations on one platform."""
    matrix = build_matrix(profile)
    if platform not in ROOTS or profile["execution"].get("authorization_id") != AUTHORIZATION:
        raise ValueError("unknown platform or authorization")
    if not profile["execution"].get("remote_submission_authorized"):
        raise ValueError("remote admission and training not authorized")
    name = profile["model"]
    training = copy.deepcopy(adaptive["training"])
    training.update(profile["training"])
    training.pop("initialization", None)
    training.pop("test_authorized", None)
    training["learning_rates"] = {name: training.pop("learning_rate")}
    runtime = copy.deepcopy(adaptive["runtime"])
    tasks = []
    for row in matrix["runs"]:
        if row["platform"] != platform or row["disposition"] != "new_training":
            continue
        variant, domain = profile["variants"][row["variant"]], row["domain"]
        spec = source["datasets"][domain]
        params = copy.deepcopy(source["models"][name])
        params.update(width=variant["width"], local_kernel=variant["local_kernel"],
                      dense_head_hidden_dim=variant["dense_head_hidden_dim"], enabled_branches=variant["branches"])
        config = dict(
            experiment=dict(experiment_id=profile["experiment_id"], authorization_id=AUTHORIZATION,
                            lane=spec["lane"], source_domain=domain, branch_variant=row["variant"],
                            held_out_test_authorized=False, primary_reporting_metric="bands_1_10_mre_pct"),
            dataset=dict(path=str(Path(ROOTS[platform]) / Path(spec["path"]).name),
                         grid_field=spec["grid_field"], expected_dataset_sha256=spec["sha256"]),
            representation=dict(resolution=64, sampling="continuous", threshold=None, physical_equivalence=True),
            models={name: params}, training=copy.deepcopy(training), runtime=copy.deepcopy(runtime),
            adaptive_optimization=copy.deepcopy(adaptive["protocol"]), output=dict(runs_dir="runs"))
        resolve_full_cell_training_config(config, model_name=name, seed=row["seed"], admission_probe=False)
        tasks.append((row, config))
    if len(tasks) != 36 or len({r["case_id"] for r, _ in tasks}) != 36:
        raise ValueError("local training matrix must contain 36 unique tasks")
    return tasks


def freeze(profile_path: Path, output: Path, platform: str, root: Path) -> None:
    """Inspect actual datasets and freeze immutable configs and task identities."""
    import torch
    from phc2d.models import build_model

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    source = yaml.safe_load((root / profile["source_profile"]).read_text(encoding="utf-8"))
    adaptive = yaml.safe_load((root / profile["adaptive_profile"]).read_text(encoding="utf-8"))
    tasks = task_configs(profile, source, adaptive, platform)
    if output.exists():
        raise FileExistsError(output)
    datasets = {}
    for row, config in tasks:
        domain = row["domain"]
        if domain not in datasets:
            spec = config["dataset"]
            datasets[domain] = inspect_dataset(dict(path=spec["path"], grid_field=spec["grid_field"],
                                                    sha256=spec["expected_dataset_sha256"]),
                                               config["training"]["train_size"], config["training"]["train_subset_salt"])
        data = datasets[domain]
        if data["target_shape"] != [81, 10] or data["validation_count"] != (1000 if domain.startswith("k5_") else 5000):
            raise ValueError("source data shape or validation count changed")
        if row["seed"] == 0:
            with torch.device("meta"):
                model = build_model(profile["model"], input_shape=tuple(data["grid_shape"]),
                                    output_shape=data["target_shape"], **config["models"][profile["model"]])
            if sum(p.numel() for p in model.parameters()) != profile["variants"][row["variant"]]["parameters"]:
                raise ValueError("actual model parameter count changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    rows = []
    for row, config in tasks:
        path = output / "configs" / f"{row['case_id']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append(dict(task_index=len(rows), case_id=row["case_id"], model=profile["model"], seed=row["seed"],
                         lane=config["experiment"]["lane"], domain=row["domain"], variant=row["variant"],
                         config=str(path), config_sha256=file_digest(path), run_dir=str(output / "training" / row["case_id"])))
    write_table(output / "task_manifest.csv", rows)
    test_rows = []
    for row in rows:
        for target in profile["test"]["targets"]:
            if target in EXTERNAL_TARGETS:
                filename, digest = EXTERNAL_TARGETS[target]
                grid_field, external = "material_area_fraction_grid_64", True
            else:
                target_spec = source["datasets"][target]
                filename, digest = Path(target_spec["path"]).name, target_spec["sha256"]
                grid_field, external = target_spec["grid_field"], False
            test_rows.append(dict(case_id=row["case_id"], target=target, dataset=str(Path(ROOTS[platform]) / filename),
                                  dataset_sha256=digest, grid_field=grid_field, external=external,
                                  checkpoint_status="awaiting_frozen_training_checkpoint"))
    write_table(output / "test_target_manifest.csv", test_rows)
    write_json(output / "datasets.json", datasets)
    write_json(output / "suite.json", dict(campaigns=[str(output)], platform=platform, gpu_limit=8,
                                           no_automatic_retry=True, test_authorized=False))
    profile["platform"] = platform
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    write_json(output / "freeze.json", dict(status="config_ready", task_count=len(rows),
               manifest_sha256=file_digest(output / "task_manifest.csv"),
               test_target_manifest_sha256=file_digest(output / "test_target_manifest.csv"),
               test_target_count=len(test_rows), profile_sha256=file_digest(profile_path),
               source_profile_sha256=file_digest(root / profile["source_profile"]),
               adaptive_profile_sha256=file_digest(root / profile["adaptive_profile"]), test_targets_loaded=False))


def audit_admission(suite: Path) -> dict:
    """Require all 12 unique local domain/variant probes to pass."""
    rows = [row for row in read_tasks(suite) if int(row["seed"]) == 0]
    if len(rows) != 12:
        raise ValueError("expected 12 local admission tasks")
    for row in rows:
        out = suite / "admission" / row["case_id"]
        result = json.loads((out / "results.json").read_text(encoding="utf-8"))
        audit = json.loads((out / "contract_audit.json").read_text(encoding="utf-8"))
        marker = json.loads((suite / "admission_claims" / (suite.name + "_" + row["case_id"]) / "completed.json").read_text())
        if (result["status"] != "full_cell_forward_admission_probe_pass" or result["test_executed"]
                or result["model"] != row["model"] or int(result["seed"]) != 0
                or result["checkpoint_sha256"] != file_digest(out / "model.pt")
                or marker["checkpoint_sha256"] != result["checkpoint_sha256"]
                or not audit["passed"] or audit["config_sha256"] != row["config_sha256"]):
            raise ValueError(f"invalid admission: {row['case_id']}")
    record = dict(passed=True, task_count=len(rows), validation_only=True,
                  manifest_sha256=file_digest(suite / "task_manifest.csv"))
    audit_path = suite / "admission_audit.json"
    if audit_path.exists():
        if json.loads(audit_path.read_text()) != record:
            raise ValueError("existing admission audit does not match current evidence")
    else:
        with audit_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    return record


def audit_release(suite: Path) -> None:
    audit = json.loads((suite / "admission_audit.json").read_text())
    if not audit["passed"] or audit["task_count"] != 12 or audit["manifest_sha256"] != file_digest(suite / "task_manifest.csv"):
        raise ValueError("local admission is not bound to this manifest")
    platform = json.loads((suite / "suite.json").read_text())["platform"]
    local_release = suite / "platform_admission_release.json"
    if local_release.is_file():
        release = json.loads(local_release.read_text())
        preflight = json.loads((suite / "manifest_preflight.json").read_text())
        baseline = json.loads((suite / "baseline_reuse_audit.json").read_text())
        if (release.get("authorization_id") != PLATFORM_RELEASE_AUTHORIZATION
                or release.get("platform") != platform or not release.get("passed")
                or release.get("manifest_sha256") != file_digest(suite / "task_manifest.csv")
                or release.get("admission_sha256") != file_digest(suite / "admission_audit.json")
                or release.get("preflight_sha256") != file_digest(suite / "manifest_preflight.json")
                or release.get("baseline_sha256") != file_digest(suite / "baseline_reuse_audit.json")
                or release.get("resource_sha256") != file_digest(suite / "resource_admission.json")
                or preflight.get("status") != "pass_training_manifest" or preflight.get("task_count") != 36
                or preflight.get("manifest_sha256") != file_digest(suite / "task_manifest.csv")
                or preflight.get("baseline_reuse_audit_sha256") != file_digest(suite / "baseline_reuse_audit.json")
                or baseline.get("status") != "pass" or baseline.get("candidate_count") != 24):
            raise ValueError("authorized local-platform release is missing or changed")
        return
    release = json.loads((suite / "global_admission_release.json").read_text())
    if (release.get("authorization_id") != AUTHORIZATION or not release.get("passed")
            or release.get("passed_tasks") != 24 or set(release.get("platforms", {})) != set(ROOTS)):
        raise ValueError("both-platform admission release is missing or changed")
    local = release["platforms"][platform]
    if (local["manifest_sha256"] != file_digest(suite / "task_manifest.csv")
            or local["admission_sha256"] != file_digest(suite / "admission_audit.json")):
        raise ValueError("both-platform admission release is missing or changed")


def release_platform(suite: Path) -> dict:
    """Bind a user-approved per-platform release to verified local evidence."""
    if (suite / "platform_admission_release.json").exists():
        raise FileExistsError(suite / "platform_admission_release.json")
    audit_admission(suite)
    preflight = json.loads((suite / "manifest_preflight.json").read_text())
    baseline = json.loads((suite / "baseline_reuse_audit.json").read_text())
    resource = json.loads((suite / "resource_admission.json").read_text())
    platform = json.loads((suite / "suite.json").read_text())["platform"]
    manifest_sha = file_digest(suite / "task_manifest.csv")
    if (platform not in ROOTS or preflight.get("status") != "pass_training_manifest"
            or preflight.get("task_count") != 36 or preflight.get("manifest_sha256") != manifest_sha
            or preflight.get("baseline_reuse_audit_sha256") != file_digest(suite / "baseline_reuse_audit.json")
            or baseline.get("status") != "pass" or baseline.get("candidate_count") != 24
            or resource.get("status") != "admission_measurements_only" or resource.get("platform") != platform
            or resource.get("admission_tasks") != 12 or resource.get("max_concurrent_gpu") != 8):
        raise ValueError("local admission, baseline or resource evidence is incomplete")
    record = dict(passed=True, platform=platform, authorization_id=PLATFORM_RELEASE_AUTHORIZATION,
                  manifest_sha256=manifest_sha, admission_sha256=file_digest(suite / "admission_audit.json"),
                  preflight_sha256=file_digest(suite / "manifest_preflight.json"),
                  baseline_sha256=file_digest(suite / "baseline_reuse_audit.json"),
                  resource_sha256=file_digest(suite / "resource_admission.json"))
    with (suite / "platform_admission_release.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    audit_release(suite)
    return record


def worker(suite: Path, stage: str) -> None:
    from phc2d.forward.full_cell import run_full_cell_forward
    if stage == "train":
        audit_release(suite)
    for row in read_tasks(suite):
        if stage == "admission" and int(row["seed"]) != 0:
            continue
        claim = claim_task(suite / f"{stage}_claims", suite.name + "_" + row["case_id"],
                           dict(job_id=os.environ.get("SLURM_JOB_ID"), task=row))
        if claim is None:
            continue
        out = suite / "admission" / row["case_id"] if stage == "admission" else Path(row["run_dir"])
        try:
            if out.exists():
                raise FileExistsError(out)
            run_full_cell_forward(row["config"], model_name=row["model"], seed=int(row["seed"]),
                                  run_dir=out, admission_probe=stage == "admission")
            audit_run(out, row, stage == "admission")
            import torch
            profile = yaml.safe_load((suite / "config.resolved.yaml").read_text(encoding="utf-8"))
            contract = json.loads((out / "contract_audit.json").read_text(encoding="utf-8"))
            dataset = json.loads((suite / "datasets.json").read_text(encoding="utf-8"))[row["domain"]]
            payload = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
            normalizer = hashlib.sha256(payload["target_mean"].numpy().tobytes() + payload["target_std"].numpy().tobytes()).hexdigest()
            if (contract["parameter_elements"] != profile["variants"][row["variant"]]["parameters"]
                    or normalizer != dataset["normalizer_hash"] or payload["k_points_sha256"] != dataset["k_points_hash"]):
                raise ValueError("trained model, normalizer or k-point identity changed")
            write_json(claim / "completed.json", dict(status="completed", checkpoint_sha256=file_digest(out / "model.pt")))
        except Exception as exc:
            write_json(claim / "failure.json", dict(error=repr(exc), retry_authorized=False))
            raise
    if stage == "train":
        from phc2d.forward.convergence_results import finish_if_complete
        finish_if_complete(suite, [str(suite)])


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["freeze", "admission", "audit-admission", "release-platform", "train"])
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--platform", choices=sorted(ROOTS))
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.profile is None or args.root is None or args.platform is None:
            parser.error("freeze requires --profile, --root and --platform")
        freeze(args.profile, args.suite, args.platform, args.root)
    elif args.stage == "audit-admission":
        print(json.dumps(audit_admission(args.suite)))
    elif args.stage == "release-platform":
        print(json.dumps(release_platform(args.suite)))
    else:
        worker(args.suite, args.stage)


if __name__ == "__main__":
    main()
