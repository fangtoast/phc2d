"""Prepare equal-count mixed-source branch campaigns without submitting jobs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import tarfile
from pathlib import Path

import yaml

from phc2d.data.full_cell_mixed import materialize_nested_mixed_validation
from phc2d.forward.budget_confirmation import file_digest, inspect_dataset, read_tasks, write_json, write_table
from phc2d.forward.full_cell_training_contracts import resolve_full_cell_training_config
from phc2d.forward.official_comparison_campaign import audit_run, claim_task
from phc2d.forward.official_comparison_test import inspect_test_target
from phc2d.forward.single_domain_branch_campaign import EXTERNAL_TARGETS, ROOTS
from phc2d.forward.single_domain_branch_matrix import build_equal_mixed_matrix


def prepare_platform(profile_path: Path, source_audit_path: Path, output: Path,
                     platform: str, root: Path, *, data_root: Path | None = None) -> dict:
    """Materialize one lane and freeze 24 configs after source identity checks.

    This stage never creates scheduler jobs or grants training/test authorization.
    An existing output directory is never reused after a failed preparation.
    """
    import torch
    from phc2d.models import build_model, describe_model_identity

    output = output.resolve()
    root = root.resolve()
    if data_root is not None:
        data_root = data_root.resolve()
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    variants = yaml.safe_load((root / profile["variant_profile"]).read_text(encoding="utf-8"))
    source = yaml.safe_load((root / profile["source_profile"]).read_text(encoding="utf-8"))
    adaptive = yaml.safe_load((root / profile["adaptive_profile"]).read_text(encoding="utf-8"))
    matrix = build_equal_mixed_matrix(profile, variants)
    if platform not in profile["platforms"] or platform not in ROOTS:
        raise ValueError("unknown platform")
    lane = profile["platforms"][platform]["lanes"][0]
    audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or audit.get("test_labels_loaded") is not False:
        raise ValueError("source selection audit is missing or inadmissible")
    if audit.get("salt") != profile["data"]["subset_salt"]:
        raise ValueError("source subset selection salt changed")
    domain_names = [f"{name}_{lane}" for name in profile["data"]["domains"]]
    base = data_root if data_root is not None else Path(ROOTS[platform])
    sources, parent, selected = {}, {}, {}
    for domain in domain_names:
        entry = audit["domains"][domain]
        expected = source["datasets"][domain]
        if entry["dataset_sha256"] != expected["sha256"]:
            raise ValueError("source audit dataset identity changed")
        evidence = entry["source_seed_evidence"]
        if (len(evidence) != 3 or {int(r["seed"]) for r in evidence} != {0, 1, 2}
                or {r["train_subset_hash"] for r in evidence} != {entry["train_parent_hash"]}):
            raise ValueError("three historical parent identities do not match")
        sources[domain] = dict(path=str(base / Path(expected["path"]).name), sha256=expected["sha256"],
                               grid_field=expected["grid_field"])
        parent[domain] = entry["train_parent_hash"]
        selected[domain] = entry["selected_train_hash"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    data = profile["data"]
    mixed = materialize_nested_mixed_validation(
        sources, output / f"mixed_{lane}.h5", parent_size=data["parent_train_per_domain"],
        selected_size=data["train_per_domain"], validation_per_domain=data["validation_per_domain"],
        salt=data["subset_salt"], expected_parent_hashes=parent, expected_selected_hashes=selected)
    dataset = inspect_dataset(mixed, profile["training"]["train_size"], data["subset_salt"])
    if dataset["train_count"] != 8000 or dataset["validation_count"] != 2000 or dataset["test_count"] != 0:
        raise ValueError("mixed data split sizes changed")
    name = variants["model"]
    training = copy.deepcopy(adaptive["training"])
    training.update(profile["training"])
    training.pop("initialization", None)
    training["learning_rates"] = {name: training.pop("learning_rate")}
    training["train_subset_salt"] = data["subset_salt"]
    model_records, rows = {}, []
    for row in matrix["runs"]:
        if row["platform"] != platform:
            continue
        variant = variants["variants"][row["variant"]]
        params = copy.deepcopy(source["models"][name])
        params.update(width=variant["width"], local_kernel=variant["local_kernel"],
                      dense_head_hidden_dim=variant["dense_head_hidden_dim"], enabled_branches=variant["branches"])
        config = dict(experiment=dict(experiment_id=profile["experiment_id"], lane=lane,
                                     source_domain=f"mixed_{lane}", branch_variant=row["variant"],
                                     held_out_test_authorized=False, primary_reporting_metric="bands_1_10_mre_pct"),
                      dataset=dict(path=mixed["path"], grid_field=mixed["grid_field"], expected_dataset_sha256=mixed["sha256"]),
                      representation=dict(resolution=dataset["grid_shape"][0], sampling="continuous", threshold=None,
                                          physical_equivalence=True),
                      models={name: params}, training=copy.deepcopy(training), runtime=copy.deepcopy(adaptive["runtime"]),
                      adaptive_optimization=copy.deepcopy(adaptive["protocol"]), output=dict(runs_dir="runs"))
        resolve_full_cell_training_config(config, model_name=name, seed=row["seed"], admission_probe=False)
        if row["variant"] not in model_records:
            with torch.device("meta"):
                model = build_model(name, input_shape=tuple(dataset["grid_shape"]),
                                    output_shape=tuple(dataset["target_shape"]), **params)
            count = sum(p.numel() for p in model.parameters())
            if count != variant["parameters"]:
                raise ValueError("model parameter count differs from frozen architecture contract")
            model_records[row["variant"]] = describe_model_identity(model, name, model_kwargs=params)
        path = output / "configs" / f"{row['case_id']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        rows.append(dict(task_index=len(rows), case_id=row["case_id"], model=name, seed=row["seed"], lane=lane,
                         domain=f"mixed_{lane}", variant=row["variant"], config=str(path),
                         config_sha256=file_digest(path), run_dir=str(output / "training" / row["case_id"])))
    targets, target_preflight = [], {}
    for row in rows:
        for target in profile["test"]["targets"]:
            if target in EXTERNAL_TARGETS:
                filename, digest = EXTERNAL_TARGETS[target]
                grid, external = "material_area_fraction_grid_64", True
            else:
                spec = source["datasets"][target]
                filename, digest = Path(spec["path"]).name, spec["sha256"]
                grid, external = spec["grid_field"], False
            if target not in target_preflight:
                checked = inspect_test_target(base / filename, dataset_sha256=digest, grid_field=grid, external=external)
                if (checked["grid_shape"] != dataset["grid_shape"] or checked["target_shape"] != dataset["target_shape"]
                        or checked["k_points_hash"] != dataset["k_points_hash"]):
                    raise ValueError(f"target input, label shape or k-point identity differs: {target}")
                target_preflight[target] = checked
            targets.append(dict(case_id=row["case_id"], target=target, dataset=str(base / filename),
                                dataset_sha256=digest, grid_field=grid, external=external,
                                checkpoint_status="awaiting_frozen_training_checkpoint"))
    if len(rows) != 24 or len(targets) != 144:
        raise ValueError("platform task or target matrix size changed")
    write_table(output / "task_manifest.csv", rows)
    write_table(output / "test_target_manifest.csv", targets)
    write_json(output / "datasets.json", {f"mixed_{lane}": dataset})
    write_json(output / "mixed_lineage.json", mixed)
    write_json(output / "model_identities.json", model_records)
    write_json(output / "target_preflight.json", dict(status="pass", targets=target_preflight, test_labels_loaded=False))
    (output / "source_selection_audit.json").write_bytes(source_audit_path.read_bytes())
    write_json(output / "suite.json", dict(campaigns=[str(output)], platform=platform, gpu_limit=8,
                                           expected_tasks=24, expected_test_cells=144,
                                           experiment_id=profile["experiment_id"], no_automatic_retry=True,
                                           train_authorized=False, test_authorized=False))
    profile["platform"] = platform
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    frozen = dict(status="data_and_configs_frozen_not_authorized", task_count=len(rows), target_count=len(targets),
                  manifest_sha256=file_digest(output / "task_manifest.csv"),
                  test_target_manifest_sha256=file_digest(output / "test_target_manifest.csv"),
                  profile_sha256=file_digest(profile_path), source_selection_audit_sha256=file_digest(source_audit_path),
                  source_profile_sha256=file_digest(root / profile["source_profile"]),
                  variant_profile_sha256=file_digest(root / profile["variant_profile"]),
                  adaptive_profile_sha256=file_digest(root / profile["adaptive_profile"]),
                  suite_sha256=file_digest(output / "suite.json"),
                  resolved_profile_sha256=file_digest(output / "config.resolved.yaml"),
                  datasets_sha256=file_digest(output / "datasets.json"),
                  lineage_sha256=file_digest(output / "mixed_lineage.json"),
                  model_identities_sha256=file_digest(output / "model_identities.json"),
                  target_preflight_sha256=file_digest(output / "target_preflight.json"),
                  mixed_dataset_sha256=mixed["sha256"], test_targets_loaded=False)
    write_json(output / "freeze.json", frozen)
    return frozen


def _verify_frozen(suite: Path) -> dict:
    frozen = json.loads((suite / "freeze.json").read_text())
    if frozen.get("status") != "data_and_configs_frozen_not_authorized":
        raise ValueError("unrecognized frozen suite")
    files = {"suite_sha256": "suite.json", "resolved_profile_sha256": "config.resolved.yaml",
             "datasets_sha256": "datasets.json", "lineage_sha256": "mixed_lineage.json",
             "model_identities_sha256": "model_identities.json", "source_selection_audit_sha256": "source_selection_audit.json",
             "test_target_manifest_sha256": "test_target_manifest.csv", "target_preflight_sha256": "target_preflight.json"}
    if any(file_digest(suite / name) != frozen[key] for key, name in files.items()):
        raise ValueError("frozen suite evidence changed")
    datasets = json.loads((suite / "datasets.json").read_text())
    if len(datasets) != 1:
        raise ValueError("one mixed dataset is required per platform")
    dataset = next(iter(datasets.values()))
    if dataset["sha256"] != frozen["mixed_dataset_sha256"] or file_digest(Path(dataset["path"])) != dataset["sha256"]:
        raise ValueError("frozen mixed data changed")
    read_tasks(suite)
    return frozen


def audit_admission(suite: Path) -> dict:
    """Check all eight local seed-zero probes before any training release."""
    if (suite / "admission_audit.json").exists() or (suite / "resource_admission.json").exists():
        raise FileExistsError(suite / "admission_audit.json")
    rows = [r for r in read_tasks(suite) if int(r["seed"]) == 0]
    if len(rows) != 8 or len({r["variant"] for r in rows}) != 8:
        raise ValueError("eight unique admission probes are required")
    resources = []
    for row in rows:
        out = suite / "admission" / row["case_id"]
        result = json.loads((out / "results.json").read_text())
        contract = json.loads((out / "contract_audit.json").read_text())
        metrics = json.loads((out / "metrics.json").read_text())
        marker = json.loads((suite / "admission_claims" / (suite.name + "_" + row["case_id"]) / "completed.json").read_text())
        digest = file_digest(out / "model.pt")
        if (result["status"] != "full_cell_forward_admission_probe_pass" or result["test_executed"]
                or not contract["passed"] or contract["config_sha256"] != row["config_sha256"]
                or result["checkpoint_sha256"] != digest or marker["checkpoint_sha256"] != digest):
            raise ValueError(f"admission identity or checkpoint mismatch: {row['case_id']}")
        fields = ("train_time_s", "eval_time_s", "peak_gpu_memory_mb", "throughput_samples_per_s")
        if (metrics["n_train"] != 8000 or metrics["n_val"] != 2000 or metrics["samples_seen"] != 64
                or metrics["optimizer_steps"] < 1 or any(not math.isfinite(float(metrics[k])) or float(metrics[k]) <= 0 for k in fields)):
            raise ValueError(f"admission resources or validation are invalid: {row['case_id']}")
        resources.append(dict(case_id=row["case_id"], **{key: metrics[key] for key in fields}))
    write_json(suite / "resource_admission.json", dict(status="admission_measurements_only", task_count=8,
        max_concurrent_gpu=8, max_total_task_epochs=24000,
        gpu_hours_estimate="not_available_single_batch_probe_is_not_full_epoch", records=resources))
    record = dict(status="admission_pass", task_count=len(rows), manifest_sha256=file_digest(suite / "task_manifest.csv"),
                  validation_only=True, test_executed=False)
    write_json(suite / "admission_audit.json", record)
    return record


def record_authorization(suite: Path, stage: str, authorization_id: str) -> dict:
    """Bind explicit user authorization to the actual frozen suite and stage."""
    if stage not in {"admission", "train", "validation", "test"} or not authorization_id.strip():
        raise ValueError("stage and explicit authorization identity are required")
    frozen = _verify_frozen(suite)
    evidence = {}
    if stage == "train":
        admission = json.loads((suite / "admission_audit.json").read_text())
        if admission["status"] != "admission_pass" or admission["task_count"] != 8 or admission["manifest_sha256"] != frozen["manifest_sha256"]:
            raise ValueError("local admission must pass before training")
        resource = json.loads((suite / "resource_admission.json").read_text())
        if resource.get("status") != "admission_measurements_only" or resource.get("task_count") != 8 or resource.get("max_concurrent_gpu") != 8:
            raise ValueError("local admission resource evidence is missing")
        evidence["admission_audit.json"] = file_digest(suite / "admission_audit.json")
        evidence["resource_admission.json"] = file_digest(suite / "resource_admission.json")
    if stage in {"validation", "test"}:
        train = json.loads((suite / "training_audit.json").read_text())
        if train.get("status") != "training_artifacts_complete_pending_independent_review" or train.get("task_count") != 24:
            raise ValueError("completed training audit required")
        evidence["training_audit.json"] = file_digest(suite / "training_audit.json")
    if stage == "test":
        from phc2d.forward.single_domain_branch_test import test_rows
        test_rows(suite)
        evidence["test_freeze.json"] = file_digest(suite / "test_freeze.json")
    record = dict(stage=stage, authorization_id=authorization_id, manifest_sha256=frozen["manifest_sha256"],
                  freeze_sha256=file_digest(suite / "freeze.json"), evidence=evidence, automatic_retry=False)
    path = suite / f"authorization_{stage}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def _check_authorization(suite: Path, stage: str) -> dict:
    _verify_frozen(suite)
    record = json.loads((suite / f"authorization_{stage}.json").read_text())
    if (record["stage"] != stage or not record["authorization_id"]
            or record["freeze_sha256"] != file_digest(suite / "freeze.json")
            or record["manifest_sha256"] != file_digest(suite / "task_manifest.csv")):
        raise ValueError("stage authorization or frozen manifest changed")
    for name, expected in record["evidence"].items():
        path = suite / name
        if file_digest(path) != expected:
            raise ValueError("authorization evidence changed")
    return record


def worker(suite: Path, stage: str) -> None:
    """Run exclusively claimed admission or training tasks on one GPU."""
    import torch
    from phc2d.forward.full_cell import run_full_cell_forward

    if stage not in {"admission", "train"}:
        raise ValueError("unsupported training worker stage")
    _check_authorization(suite, stage)
    rows = read_tasks(suite)
    identities = json.loads((suite / "model_identities.json").read_text())
    datasets = json.loads((suite / "datasets.json").read_text())
    for row in rows:
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
            data = datasets[row["domain"]]
            if file_digest(Path(data["path"])) != data["sha256"]:
                raise ValueError("mixed source dataset changed")
            run_full_cell_forward(row["config"], model_name=row["model"], seed=int(row["seed"]),
                                  run_dir=out, admission_probe=stage == "admission")
            audit_run(out, row, stage == "admission")
            contract = json.loads((out / "contract_audit.json").read_text())
            result = json.loads((out / "results.json").read_text())
            payload = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
            normalizer = hashlib.sha256(payload["target_mean"].numpy().tobytes() + payload["target_std"].numpy().tobytes()).hexdigest()
            identity, data = identities[row["variant"]], datasets[row["domain"]]
            if (contract["parameter_elements"] != identity["parameter_count"] or normalizer != data["normalizer_hash"]
                    or payload["k_points_sha256"] != data["k_points_hash"]
                    or result["model_identity"]["architecture_fingerprint"] != identity["architecture_fingerprint"]
                    or result["model_identity"]["state_dict_schema_hash"] != identity["state_dict_schema_hash"]):
                raise ValueError("model identity, normalizer or k-point identity changed")
            write_json(claim / "completed.json", dict(status="completed", checkpoint_sha256=file_digest(out / "model.pt")))
        except Exception as exc:
            write_json(claim / "failure.json", dict(error=repr(exc), retry_authorized=False))
            raise
    if stage == "train":
        from phc2d.forward.convergence_results import finish_if_complete
        finish_if_complete(suite, [str(suite)])


def submit(suite: Path, root: Path, stage: str, environment: Path) -> dict:
    """Submit one array after explicit authorization; never retry an intent."""
    if stage not in {"admission", "train", "validation", "test"}:
        raise ValueError("unsupported scheduler stage")
    auth = _check_authorization(suite, stage)
    info = json.loads((suite / "suite.json").read_text())
    profile = yaml.safe_load((suite / "config.resolved.yaml").read_text())
    platform = info["platform"]
    settings = profile["platforms"][platform]
    if settings["max_gpus"] != 8:
        raise ValueError("platform GPU limit changed")
    # Refuse overlap with any previously submitted stage in this suite.
    live = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"],
                          capture_output=True, text=True, check=True).stdout.splitlines()
    for path in suite.glob("submission_*.json"):
        old = json.loads(path.read_text())
        if old.get("status") in {"submission_intent", "submission_failed"}:
            raise ValueError("earlier submission is failed or uncertain; inspect manually")
        if any(line.strip().split("_", 1)[0] == str(old.get("job_id")) for line in live):
            raise ValueError("a previous suite stage still owns GPU resources")
    deployment = json.loads((root / "deployment_audit.json").read_text())
    launcher = root / "jobs/paracloud_equal_mixed_branch.sh"
    if file_digest(root / "deployment.tar.gz") != deployment["archive_sha256"] or not launcher.is_file():
        raise ValueError("deployment archive or launcher is invalid")
    with tarfile.open(root / "deployment.tar.gz", "r:gz") as archive:
        bundled = archive.extractfile("jobs/paracloud_equal_mixed_branch.sh")
        if bundled is None or hashlib.sha256(bundled.read()).hexdigest() != file_digest(launcher):
            raise ValueError("launcher differs from frozen deployment archive")
    marker = suite / f"submission_{stage}.json"
    export = dict(PHC2D_ROOT=str(root), PHC2D_SUITE=str(suite), PHC2D_STAGE=stage,
                  PHC2D_GPU_PLATFORM=platform, PHC2D_ENV=str(environment),
                  PHC2D_ARCHIVE_SHA256=deployment["archive_sha256"], PYTHONDONTWRITEBYTECODE="1")
    args = ["sbatch", "--parsable", "--no-requeue", "-p", settings["partition"], "--gres=gpu:1",
            "-c", str(settings["cpus_per_gpu"]), "-t", "04:00:00" if stage in {"admission", "validation"} else "7-00:00:00",
            "-J", "phc2d033_" + stage, "--array=0-7%8", "-o", str(suite / (stage + "_%A_%a.out")),
            "-e", str(suite / (stage + "_%A_%a.err")),
            "--export=ALL," + ",".join(k + "=" + v for k, v in export.items()), str(launcher)]
    record = dict(status="submission_intent", stage=stage, authorization_id=auth["authorization_id"],
                  authorization_sha256=file_digest(suite / f"authorization_{stage}.json"), command=args,
                  archive_sha256=deployment["archive_sha256"], launcher_sha256=file_digest(launcher), automatic_retry=False)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    result = subprocess.run(args, capture_output=True, text=True, cwd=root)
    record.update(status="submitted" if result.returncode == 0 else "submission_failed",
                  returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0:
        record["job_id"] = result.stdout.strip().split(";")[0]
    write_json(marker, record)
    if result.returncode:
        raise RuntimeError("scheduler submission failed; no automatic retry")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "authorize", "admission", "audit-admission", "train", "submit"])
    for name in ("profile", "source-audit", "output", "root", "suite", "environment"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--platform", choices=sorted(ROOTS))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--worker-stage", choices=["admission", "train", "validation", "test"])
    parser.add_argument("--authorization-id")
    args = parser.parse_args()
    if args.stage == "prepare":
        if any(x is None for x in (args.profile, args.source_audit, args.output, args.platform, args.root)):
            parser.error("prepare requires profile, source-audit, output, platform and root")
        result = prepare_platform(args.profile, args.source_audit, args.output, args.platform, args.root, data_root=args.data_root)
    else:
        if args.suite is None:
            parser.error("this stage requires --suite")
        if args.stage == "authorize":
            result = record_authorization(args.suite, args.worker_stage or "", args.authorization_id or "")
        elif args.stage == "audit-admission":
            result = audit_admission(args.suite)
        elif args.stage == "submit":
            if args.root is None or args.environment is None or args.worker_stage is None:
                parser.error("submit requires root, environment and worker-stage")
            result = submit(args.suite, args.root, args.worker_stage, args.environment)
        else:
            worker(args.suite, args.stage)
            return
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
