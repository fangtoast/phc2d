"""Frozen target and checkpoint contracts for EXP-FWD-032 tests.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
import csv
import hashlib
import json

import numpy as np
import pytest

from phc2d.forward.budget_confirmation import file_digest
from phc2d.forward.official_comparison_test import evaluation_role
from phc2d.forward.single_domain_branch_test import freeze_test, test_rows as read_test_rows


def test_mixed_source_roles_are_not_mislabeled_as_unseen_domain():
    assert evaluation_role("mixed_binary", "k5_binary", external=False) == "seen_source_domain_test"
    assert evaluation_role("mixed_binary", "k20_binary", external=False) == "seen_source_domain_test"
    assert evaluation_role("mixed_binary", "k5_graded", external=False) == "cross_domain_test"
    assert evaluation_role("mixed_binary", "f50_binary", external=False) == "cross_domain_test"
    assert evaluation_role("mixed_binary", "exp_dat_007_12k", external=True) == "new_geometry_transfer"
    assert evaluation_role("k5_binary", "k5_binary", external=False) == "native_test"


@pytest.mark.parametrize(("experiment", "domains", "expected"), [
    ("EXP-FWD-032", ("k5_binary", "f50_graded"), 216),
    ("EXP-FWD-033", ("mixed_binary",), 144),
])
def test_freeze_requires_authorization_and_binds_target_split(tmp_path, monkeypatch, experiment, domains, expected):
    h5py = pytest.importorskip("h5py")
    from phc2d.forward import single_domain_branch_test as module

    tasks = []
    for domain in domains:
        for variant in range(6 if experiment == "EXP-FWD-032" else 8):
            for seed in range(3):
                tasks.append(dict(case_id=f"{domain}_v{variant}_s{seed}", domain=domain, variant=str(variant),
                                  seed=seed, model="model", run_dir=str(tmp_path / "training" / f"{domain}_{variant}_{seed}")))
    monkeypatch.setattr(module, "read_tasks", lambda _: tasks)
    monkeypatch.setattr(module, "ready_checkpoint", lambda *a: {"checkpoint_sha256": "checkpoint"})
    points = np.arange(162, dtype=np.float32).reshape(81, 2)
    k_hash = hashlib.sha256(points.tobytes()).hexdigest()
    (tmp_path / "datasets.json").write_text(json.dumps({d: {"k_points_hash": k_hash, "grid_shape": [64, 64], "target_shape": [81, 10]} for d in domains}))
    (tmp_path / "training_audit.json").write_text(json.dumps(dict(task_count=len(tasks), status="training_artifacts_complete_pending_independent_review")))
    (tmp_path / "suite.json").write_text(json.dumps(dict(experiment_id=experiment, expected_tasks=len(tasks), expected_test_cells=expected)))
    targets = []
    for i in range(6):
        path = tmp_path / f"data{i}.h5"
        with h5py.File(path, "w") as handle:
            handle["grid"] = np.zeros((3, 64, 64), dtype=np.float32)
            handle["Wh"] = np.zeros((3, 81, 10), dtype=np.float32)
            handle["k_points"] = points
            handle["idx_train"] = [0]
            handle["idx_val"] = [1]
            handle["idx_test"] = [2]
        for task in tasks:
            targets.append(dict(case_id=task["case_id"], target=f"target{i}", dataset=str(path),
                                dataset_sha256=file_digest(path), grid_field="grid", external=False))
    plan = tmp_path / "test_target_manifest.csv"
    with plan.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    (tmp_path / "freeze.json").write_text(json.dumps(dict(test_target_manifest_sha256=file_digest(plan))))
    with pytest.raises(ValueError, match="authorization"):
        freeze_test(tmp_path, "")
    freeze_test(tmp_path, "test-authorized")
    rows = read_test_rows(tmp_path)
    assert len(rows) == expected
    assert all(row["checkpoint_sha256"] == "checkpoint" and int(row["sample_count"]) == 1 for row in rows)
    with (tmp_path / "test_manifest.csv").open("a") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="manifest changed"):
        read_test_rows(tmp_path)


def test_test_worker_never_repeats_a_completed_claim(tmp_path, monkeypatch):
    from phc2d.forward import single_domain_branch_test as module
    run = tmp_path / "training"
    run.mkdir()
    (run / "model.pt").write_bytes(b"weights")
    row = dict(case_id="case_s0", target="target", source_run=str(run), checkpoint_sha256=file_digest(run / "model.pt"),
               dataset="data.h5", dataset_sha256="data", grid_field="grid", external="False",
               result_dir=str(tmp_path / "test"), split_sha256="indices", sample_count="1")
    monkeypatch.setattr(module, "test_rows", lambda _: [row])
    calls = []
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: (calls.append(k) or
                        dict(target_split_sha256="indices", sample_count=1)))
    module.test_worker(tmp_path)
    module.test_worker(tmp_path)
    assert len(calls) == 1 and calls[0]["expected_experiment"] == "EXP-FWD-032"
    assert calls[0]["save_predictions"]
