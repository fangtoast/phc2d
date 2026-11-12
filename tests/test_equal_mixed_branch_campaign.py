"""Contracts for equal-count mixed-source campaign preparation and gates.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from phc2d.forward.budget_confirmation import file_digest, read_tasks
from phc2d.forward.equal_mixed_branch_campaign import prepare_platform, record_authorization, submit

ROOT = Path.cwd()
PROFILE = ROOT / "configs/research/full_cell_equal_mixed_branch_exp_fwd_033_v1.yaml"


def source_audit(path):
    source = yaml.safe_load((ROOT / "configs/research/full_cell_core_confirmation_exp_fwd_024_v1.yaml").read_text())
    domains = {}
    for name in ("k5_binary", "k20_binary", "k5_graded", "k20_graded"):
        parent = name + "-parent"
        domains[name] = dict(dataset_sha256=source["datasets"][name]["sha256"], train_parent_hash=parent,
                             selected_train_hash=name + "-selected",
                             source_seed_evidence=[dict(seed=seed, train_subset_hash=parent) for seed in range(3)])
    path.write_text(json.dumps(dict(status="pass", test_labels_loaded=False,
                                    salt="exp-fwd-021-domain-ranking-v1", domains=domains)))
    return path


def test_preparation_freezes_24_configs_and_refuses_training_without_admission(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from phc2d.forward import equal_mixed_branch_campaign as campaign

    def materialize(sources, output, **kwargs):
        output.write_bytes(b"mixed-fixture")
        assert set(sources) == {"k5_binary", "k20_binary"}
        assert kwargs["parent_size"] == 8000 and kwargs["selected_size"] == 4000
        return dict(path=str(output), sha256=file_digest(output), grid_field="input_grid", lineage=[])

    monkeypatch.setattr(campaign, "materialize_nested_mixed_validation", materialize)
    monkeypatch.setattr(campaign, "inspect_dataset", lambda spec, *a: dict(path=spec["path"], sha256=spec["sha256"],
        train_count=8000, validation_count=2000, test_count=0, grid_shape=[64, 64], target_shape=[81, 10],
        normalizer_hash="normalizer", k_points_hash="points"))
    monkeypatch.setattr(campaign, "inspect_test_target", lambda *a, **k: dict(grid_shape=[64, 64],
        target_shape=[81, 10], k_points_hash="points", dataset_sha256=k["dataset_sha256"]))
    suite = tmp_path / "suite"
    frozen = prepare_platform(PROFILE, source_audit(tmp_path / "source.json"), suite, "n40", ROOT, data_root=tmp_path)
    assert frozen["task_count"] == 24 and frozen["target_count"] == 144
    rows = read_tasks(suite)
    assert len(rows) == 24 and len({r["case_id"] for r in rows}) == 24
    assert all(r["lane"] == "binary" and r["domain"] == "mixed_binary" for r in rows)
    identity = json.loads((suite / "model_identities.json").read_text())
    assert len(identity) == 8 and all(v["architecture_fingerprint"] and v["state_dict_schema_hash"] for v in identity.values())
    assert json.loads((suite / "suite.json").read_text())["train_authorized"] is False
    with pytest.raises(FileNotFoundError):
        record_authorization(suite, "train", "user-approval")
    record = record_authorization(suite, "admission", "user-approval")
    assert record["manifest_sha256"] == frozen["manifest_sha256"]
    with pytest.raises(FileExistsError):
        record_authorization(suite, "admission", "user-approval")
    with pytest.raises(FileExistsError):
        prepare_platform(PROFILE, tmp_path / "source.json", suite, "n40", ROOT, data_root=tmp_path)


def test_preparation_rejects_mismatched_parent_seed_evidence_before_output(tmp_path):
    audit = source_audit(tmp_path / "source.json")
    content = json.loads(audit.read_text())
    content["domains"]["k5_binary"]["source_seed_evidence"][1]["train_subset_hash"] = "different"
    audit.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="parent identities"):
        prepare_platform(PROFILE, audit, tmp_path / "suite", "n40", ROOT, data_root=tmp_path)
    assert not (tmp_path / "suite").exists()


def test_submission_is_authorized_exclusive_and_preserves_failed_intent(tmp_path, monkeypatch):
    suite, root = tmp_path / "suite", tmp_path / "deploy"
    suite.mkdir()
    (root / "jobs").mkdir(parents=True)
    launcher = root / "jobs/paracloud_equal_mixed_branch.sh"
    launcher.write_text("#!/bin/bash\nexit 0\n")
    with tarfile.open(root / "deployment.tar.gz", "w:gz") as archive:
        archive.add(launcher, arcname="jobs/paracloud_equal_mixed_branch.sh")
    (root / "deployment_audit.json").write_text(json.dumps(dict(archive_sha256=file_digest(root / "deployment.tar.gz"))))
    (suite / "task_manifest.csv").write_text("case_id\ntest\n")
    config = suite / "config.yaml"
    config.write_text("{}")
    (suite / "task_manifest.csv").write_text(f"case_id,config,config_sha256\ntest,{config},{file_digest(config)}\n")
    (suite / "suite.json").write_text(json.dumps(dict(platform="n40")))
    (suite / "config.resolved.yaml").write_text(yaml.safe_dump(dict(platforms={"n40": dict(max_gpus=8,
        partition="gpu_4090", cpus_per_gpu=6)})))
    data = suite / "mixed.h5"
    data.write_bytes(b"data")
    (suite / "datasets.json").write_text(json.dumps(dict(mixed_binary=dict(path=str(data), sha256=file_digest(data)))))
    for name in ("mixed_lineage.json", "model_identities.json", "source_selection_audit.json", "test_target_manifest.csv", "target_preflight.json"):
        (suite / name).write_text("{}")
    fields = {"suite_sha256": "suite.json", "resolved_profile_sha256": "config.resolved.yaml",
              "datasets_sha256": "datasets.json", "lineage_sha256": "mixed_lineage.json",
              "model_identities_sha256": "model_identities.json", "source_selection_audit_sha256": "source_selection_audit.json",
              "test_target_manifest_sha256": "test_target_manifest.csv", "target_preflight_sha256": "target_preflight.json"}
    (suite / "freeze.json").write_text(json.dumps(dict(status="data_and_configs_frozen_not_authorized",
        manifest_sha256=file_digest(suite / "task_manifest.csv"), mixed_dataset_sha256=file_digest(data),
        **{key: file_digest(suite / name) for key, name in fields.items()})))
    with pytest.raises(FileNotFoundError):
        submit(suite, root, "admission", tmp_path / "env")
    (suite / "authorization_admission.json").write_text(json.dumps(dict(stage="admission", authorization_id="approved",
        manifest_sha256=file_digest(suite / "task_manifest.csv"), freeze_sha256=file_digest(suite / "freeze.json"), evidence={})))
    monkeypatch.setenv("USER", "test-user")
    calls = []
    def scheduler(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="" if args[0] == "squeue" else "12345\n", stderr="")
    monkeypatch.setattr("subprocess.run", scheduler)
    assert submit(suite, root, "admission", tmp_path / "env")["job_id"] == "12345"
    assert "--array=0-7%8" in calls[-1] and "--no-requeue" in calls[-1]
    with pytest.raises(FileExistsError):
        submit(suite, root, "admission", tmp_path / "env")
    assert sum(call[0] == "sbatch" for call in calls) == 1
