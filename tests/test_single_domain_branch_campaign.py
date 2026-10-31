"""Tests for the EXP-FWD-032 training campaign contract.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import copy
import json
from pathlib import Path

import pytest
import yaml

from phc2d.forward.single_domain_branch_campaign import freeze, task_configs, worker


def profiles():
    profile = yaml.safe_load(Path("configs/research/full_cell_single_domain_branch_ablation_exp_fwd_032_v1.yaml").read_text())
    source = yaml.safe_load(Path(profile["source_profile"]).read_text())
    adaptive = yaml.safe_load(Path(profile["adaptive_profile"]).read_text())
    return profile, source, adaptive


def test_both_partitions_preserve_full_training_contract():
    profile, source, adaptive = profiles()
    cases = []
    for platform in ("n40", "n56"):
        tasks = task_configs(profile, source, adaptive, platform)
        assert len(tasks) == 36
        assert sum(row["seed"] == 0 for row, _ in tasks) == 12
        for row, config in tasks:
            cases.append(row["case_id"])
            assert config["experiment"]["experiment_id"] == "EXP-FWD-032"
            assert config["training"]["micro_batch_size"] == config["training"]["effective_batch_size"] == 64
            assert config["training"]["checkpoint_selection"] == "validation_bands_2_10_mre"
            assert config["training"]["max_epochs"] == 1000
            assert config["adaptive_optimization"] == adaptive["protocol"]
            assert not config.get("continuation")
    assert len(set(cases)) == 72


def test_unauthorized_and_training_without_global_release_fail(tmp_path):
    profile, source, adaptive = profiles()
    bad = copy.deepcopy(profile)
    bad["execution"]["remote_submission_authorized"] = False
    with pytest.raises(ValueError, match="not authorized"):
        task_configs(bad, source, adaptive, "n40")
    (tmp_path / "task_manifest.csv").write_text("frozen")
    from phc2d.forward.budget_confirmation import file_digest
    (tmp_path / "admission_audit.json").write_text(json.dumps(dict(
        passed=True, task_count=12, manifest_sha256=file_digest(tmp_path / "task_manifest.csv"))))
    with pytest.raises(FileNotFoundError):
        worker(tmp_path, "train")


def test_freeze_binds_config_and_rejects_existing_output(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from phc2d.forward import single_domain_branch_campaign as campaign
    from phc2d.forward.budget_confirmation import read_tasks

    def inspect(spec, *_):
        return dict(grid_shape=[64, 64], target_shape=[81, 10],
                    validation_count=1000 if "k5" in spec["path"] else 5000)
    monkeypatch.setattr(campaign, "inspect_dataset", inspect)
    profile = Path("configs/research/full_cell_single_domain_branch_ablation_exp_fwd_032_v1.yaml")
    suite = tmp_path / "suite"
    freeze(profile, suite, "n40", Path.cwd())
    assert len(read_tasks(suite)) == 36
    import csv
    with (suite / "test_target_manifest.csv").open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    assert len(cells) == len({(r["case_id"], r["target"]) for r in cells}) == 216
    assert all(r["checkpoint_status"] == "awaiting_frozen_training_checkpoint" for r in cells)
    first = read_tasks(suite)[0]
    Path(first["config"]).write_text("tampered")
    with pytest.raises(ValueError, match="config changed"):
        read_tasks(suite)
    with pytest.raises(FileExistsError):
        freeze(profile, suite, "n40", Path.cwd())


def test_submission_is_bounded_and_never_repeated(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("submit_branch", "jobs/paracloud_submit_single_domain_branch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile, _, _ = profiles()
    profile["platform"] = "n40"
    (tmp_path / "config.resolved.yaml").write_text(yaml.safe_dump(profile))
    (tmp_path / "deployment_audit.json").write_text(json.dumps(dict(archive_sha256="frozen")))
    from phc2d.forward.budget_confirmation import file_digest
    (tmp_path / "jobs").mkdir()
    launcher = tmp_path / "jobs/paracloud_single_domain_branch.sh"
    launcher.write_text("launcher")
    (tmp_path / "launcher_audit.json").write_text(json.dumps(dict(archive_sha256="frozen", launcher_sha256=file_digest(launcher))))
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw: (calls.append(args) or
                        SimpleNamespace(returncode=0, stdout="12345\n", stderr="")))
    record = module.submit(tmp_path, tmp_path, "admission")
    assert record["job_id"] == "12345"
    assert "--array=0-7%8" in calls[0] and "--no-requeue" in calls[0]
    with pytest.raises(FileExistsError):
        module.submit(tmp_path, tmp_path, "admission")
    assert len(calls) == 1


def test_submission_recovery_requires_explicit_authorization_and_failed_prior(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    from phc2d.forward.budget_confirmation import file_digest
    spec = importlib.util.spec_from_file_location("submit_branch", "jobs/paracloud_submit_single_domain_branch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile, _, _ = profiles()
    profile["platform"] = "n40"
    (tmp_path / "config.resolved.yaml").write_text(yaml.safe_dump(profile))
    (tmp_path / "deployment_audit.json").write_text(json.dumps(dict(archive_sha256="frozen")))
    (tmp_path / "jobs").mkdir()
    launcher = tmp_path / "jobs/paracloud_single_domain_branch.sh"
    launcher.write_text("launcher")
    (tmp_path / "launcher_audit.json").write_text(json.dumps(dict(archive_sha256="frozen", launcher_sha256=file_digest(launcher))))
    (tmp_path / "submission_admission.json").write_text(json.dumps(dict(status="submission_failed", stdout="")))
    with pytest.raises(ValueError, match="explicit authorization"):
        module.submit(tmp_path, tmp_path, "admission", recovery_id="1")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="123\n", stderr=""))
    module.submit(tmp_path, tmp_path, "admission", recovery_id="1", recovery_authorization="user-approved")
    with pytest.raises(ValueError, match="may have created a job"):
        module.submit(tmp_path, tmp_path, "admission", recovery_id="2", recovery_authorization="user-approved")


def test_validation_submission_checks_live_queue_without_querying_expired_job(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    from phc2d.forward.budget_confirmation import file_digest
    spec = importlib.util.spec_from_file_location("submit_branch", "jobs/paracloud_submit_single_domain_branch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile, _, _ = profiles()
    profile["platform"] = "n40"
    (tmp_path / "config.resolved.yaml").write_text(yaml.safe_dump(profile))
    (tmp_path / "training_audit.json").write_text(json.dumps(dict(task_count=36, status="training_artifacts_complete_pending_independent_review")))
    (tmp_path / "submission_train.json").write_text(json.dumps(dict(status="submitted", job_id="123")))
    (tmp_path / "deployment_audit.json").write_text(json.dumps(dict(archive_sha256="frozen")))
    (tmp_path / "jobs").mkdir()
    launcher = tmp_path / "jobs/paracloud_single_domain_branch.sh"
    launcher.write_text("launcher")
    (tmp_path / "launcher_audit.json").write_text(json.dumps(dict(archive_sha256="frozen", launcher_sha256=file_digest(launcher))))
    monkeypatch.setenv("USER", "test-user")
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw: SimpleNamespace(
        returncode=0, stdout="123_[0-7%8]\n" if args[0] == "squeue" else "456\n", stderr=""))
    with pytest.raises(ValueError, match="not released"):
        module.submit(tmp_path, tmp_path, "validation")
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="999_0\n" if args[0] == "squeue" else "456\n", stderr="")
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.submit(tmp_path, tmp_path, "validation")["job_id"] == "456"
    assert calls[0] == ["squeue", "-h", "-u", "test-user", "-o", "%i"]
