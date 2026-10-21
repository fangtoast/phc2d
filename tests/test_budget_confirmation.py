"""Verify fixed-budget confirmation boundaries and immutable input freezes.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from pathlib import Path
import copy
import json

import numpy as np
import pytest
import yaml

from phc2d.forward.budget_confirmation import file_digest, freeze_campaign, freeze_native_test, inspect_dataset, read_tasks
from phc2d.forward.full_cell_training_contracts import resolve_full_cell_training_config


def test_registered_300_epoch_contract_preserves_legacy_guardrails():
    profile = yaml.safe_load(Path("configs/research/full_cell_core_confirmation_exp_fwd_024_v1.yaml").read_text())
    model = "global_local_band_operator_v1"
    config = {"experiment": {"experiment_id": "EXP-FWD-024"}, "models": profile["models"],
              "training": profile["training"], "runtime": profile["runtime"]}
    result = resolve_full_cell_training_config(config, model_name=model, seed=2, admission_probe=False)
    assert result["training"]["max_epochs"] == 300
    assert not result["continuation"]
    for experiment in ("EXP-FWD-022", "unregistered"):
        legacy = copy.deepcopy(config)
        legacy["experiment"]["experiment_id"] = experiment
        with pytest.raises(ValueError, match="max_epochs"):
            resolve_full_cell_training_config(legacy, model_name=model, seed=2, admission_probe=False)


def make_dataset(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as h:
        h["grid"] = np.zeros((12, 4, 4), dtype=np.float32)
        labels = np.ones((12, 3, 2), dtype=np.float32)
        labels[-2:] = np.nan  # The freeze must not read or normalize test labels.
        h["Wh"] = labels
        h["idx_train"] = np.arange(8)
        h["idx_val"] = np.arange(8, 10)
        h["idx_test"] = np.arange(10, 12)
        h["sample_id"] = np.array([f"s{i}" for i in range(12)], dtype="S")
        h["k_points"] = np.zeros((3, 2), dtype=np.float32)
    return {"path": str(path), "sha256": file_digest(path), "grid_field": "grid", "lane": "binary"}


def test_freeze_rejects_overlap_and_corruption_without_reading_test_labels(tmp_path):
    h5py = pytest.importorskip("h5py")
    spec = make_dataset(tmp_path)
    record = inspect_dataset(spec, 6, "frozen")
    assert record["train_count"] == 6 and not record["test_targets_loaded"]
    with h5py.File(spec["path"], "r+") as h:
        h["idx_test"][0] = 0
    with pytest.raises(ValueError, match="SHA-256"):
        inspect_dataset(spec, 6, "frozen")
    spec["sha256"] = file_digest(Path(spec["path"]))
    with pytest.raises(ValueError, match="overlap"):
        inspect_dataset(spec, 6, "frozen")


def test_freeze_produces_paired_matrix_and_refuses_mutated_configs(tmp_path):
    spec = make_dataset(tmp_path)
    profile = yaml.safe_load(Path("configs/research/full_cell_core_confirmation_exp_fwd_024_v1.yaml").read_text())
    profile["training"]["train_size"] = 6
    profile["datasets"] = {name: spec for name in profile["datasets"]}
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile))
    campaign = tmp_path / "campaign"
    freeze_campaign(path, campaign)
    rows = read_tasks(campaign)
    assert len(rows) == 54 and len({r["case_id"] for r in rows}) == 54
    assert {int(r["seed"]) for r in rows} == {0, 1, 2}
    assert not json.loads((campaign / "freeze.json").read_text())["test_targets_loaded"]
    with pytest.raises(FileExistsError):
        freeze_campaign(path, campaign)
    Path(rows[0]["config"]).write_text("modified")
    with pytest.raises(ValueError, match="config changed"):
        read_tasks(campaign)


def test_branch_test_remains_locked(tmp_path):
    (tmp_path / "config.resolved.yaml").write_text("experiment_id: EXP-FWD-025\n")
    with pytest.raises(ValueError, match="not authorized"):
        freeze_native_test(tmp_path)


def test_execution_overlay_preserves_scientific_contract(tmp_path):
    spec = make_dataset(tmp_path)
    base = yaml.safe_load(Path("configs/research/full_cell_core_confirmation_exp_fwd_024_v1.yaml").read_text())
    base["training"]["train_size"] = 6
    base["datasets"] = {key: spec for key in base["datasets"]}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    overlay = {"base_profile": "base.yaml", "dataset_path_root": str(tmp_path),
               "execution_platform_basis": "explicit_n40_or_n56_authorization", "platform": {"host": "paracloud-n40"}}
    path = tmp_path / "overlay.yaml"
    path.write_text(yaml.safe_dump(overlay))
    freeze_campaign(path, tmp_path / "campaign")
    result = yaml.safe_load((tmp_path / "campaign/config.resolved.yaml").read_text())
    assert result["training"] == base["training"] and result["models"] == base["models"]
    assert result["seeds"] == [0, 1, 2] and result["platform"]["host"] == "paracloud-n40"
    assert result["base_profile_sha256"] == file_digest(tmp_path / "base.yaml")


def test_native_test_waits_for_every_bundle_and_finalizes_once(tmp_path, monkeypatch):
    import phc2d.forward.budget_confirmation as module
    (tmp_path / "config.resolved.yaml").write_text("experiment_id: EXP-FWD-024\n")
    (tmp_path / "freeze.json").write_text(json.dumps({"manifest_sha256": "frozen"}))
    calls = []
    monkeypatch.setattr(module, "freeze_native_test", lambda p: calls.append("freeze"))
    monkeypatch.setattr(module, "run_test_bundle", lambda *args: calls.append("test"))
    monkeypatch.setattr(module, "summarize_native_test", lambda p: calls.append("summary"))
    module.finish_native_when_complete(tmp_path, 0, 2)
    assert calls == []
    module.finish_native_when_complete(tmp_path, 1, 2)
    assert calls == ["freeze", "test", "summary"]
    assert (tmp_path / "native_test_finalizer_claim.json").is_file()
    with pytest.raises(FileExistsError):
        module.finish_native_when_complete(tmp_path, 1, 2)
