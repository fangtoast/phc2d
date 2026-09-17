"""Isolation, identity, equal-domain aggregation and Draft reporting contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.forward.full_cell_multidomain_contracts import read_csv, sha256_array, sha256_file, write_csv
from phc2d.forward.full_cell_transfer_contracts import audit_transfer_cohort, checkpoint_identity_map
from phc2d.forward.full_cell_transfer_summary import merge_domain_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_transfer_cli_help_does_not_import_training_or_solver_stacks() -> None:
    program = "from phc2d.forward.full_cell_transfer_campaign import main\nimport sys\ntry: main(['--help'])\nexcept SystemExit: pass\nassert not {'torch','h5py','meep'} & set(sys.modules)"
    result = subprocess.run([sys.executable, "-c", program], env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_profile_is_63_fixed_checkpoints_and_six_disjoint_new_targets() -> None:
    config = yaml.safe_load((ROOT / "configs/research/full_cell_transfer_exp_fwd_023_v1.yaml").read_text())
    assert (config["checkpoint_count"], config["new_cell_count"], config["combined_cell_count"]) == (63, 378, 756)
    assert config["dataset_acceptance"] == "cluster_audited_pending_wsl_acceptance"
    assert set(config["datasets"]) == {f"k{k}_{lane}" for k in (8, 10, 15) for lane in ("binary", "graded")}
    for domain, row in config["datasets"].items():
        assert row["test_count"] == 1000 and row["train_count"] == 8000
        assert len(row["sha256"]) == 64 and row["path"].startswith("/data/run01/scxl892/")
        assert row["grid_field"] == ("material_area_fraction_grid_64" if domain.endswith("binary") else "material_weight_grid_64")


def _matrix():
    checkpoints = [{"regime": "S1-B", "model": model} for model in ("a", "b")]
    targets = ["k5_binary", "f50_binary"]
    rows = [{**cp, "target_domain": target, "sample_count": 1000 if target.startswith("k5") else 5000,
             "band1_mre_pct": value, "bands_2_10_mre_pct": value, "bands_1_10_mre_pct": value}
            for cp in checkpoints for target, value in zip(targets, (1.0, 9.0))]
    return checkpoints, targets, rows


def test_domain_macro_is_not_weighted_by_f50_sample_count() -> None:
    checkpoints, targets, rows = _matrix()
    matrix, macros = merge_domain_matrix(rows, checkpoints, targets)
    assert len(matrix) == 4 and len(macros) == 2
    assert all(r["bands_2_10_mre_pct"] == 5.0 for r in macros)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unexpected", "nonfinite"])
def test_matrix_fails_closed_on_integrity_defects(defect: str) -> None:
    checkpoints, targets, rows = _matrix()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "unexpected":
        rows[0] = {**rows[0], "target_domain": "other"}
    else:
        rows[0] = {**rows[0], "band1_mre_pct": float("nan")}
    with pytest.raises(RuntimeError):
        merge_domain_matrix(rows, checkpoints, targets)


def test_checkpoint_identity_requires_unique_identity_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "evaluation/checkpoint_manifest.csv"
    rows = [{"task_id": "a", "checkpoint_sha256": "sha-a"}, {"task_id": "b", "checkpoint_sha256": "sha-b"}]
    write_csv(path, rows)
    assert checkpoint_identity_map(tmp_path) == {"a": "sha-a", "b": "sha-b"}
    for defect in ({**rows[1], "task_id": "a"}, {**rows[1], "checkpoint_sha256": "sha-a"}):
        write_csv(path, [rows[0], defect])
        with pytest.raises(RuntimeError, match="collision"):
            checkpoint_identity_map(tmp_path)


@pytest.fixture
def cohort(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "source"
    source.mkdir()
    k = np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    path = source / "source.h5"
    with h5py.File(path, "w") as f:
        f["grid"] = np.stack([np.full((2, 3), i + 1, dtype=np.float32) for i in range(3)])
        f["sample_id"] = np.asarray([b"source-train", b"source-val", b"source-test"])
        f["idx_val"] = np.asarray([1], dtype=np.int64)
    # No Wh dataset exists: the leakage auditor must not read any labels.
    indices = np.asarray([0], dtype=np.int64)
    np.save(source / "selected.npy", indices)
    write_csv(source / "subset_manifest.csv", [{"domain": "source_binary", "indices_path": "selected.npy", "indices_sha256": sha256_array(indices, dtype=np.int64)}])
    write_csv(source / "dataset_manifest.csv", [{"domain": "source_binary", "dataset_path": str(path), "dataset_sha256": sha256_file(path), "grid_field": "grid",
               "idx_val_hash": sha256_array(np.asarray([1]), dtype=np.int64), "k_points_sha256": sha256_array(k, dtype=np.float32)}])
    targets = {}
    for cutoff in (8, 10, 15):
        p = tmp_path / f"k{cutoff}.h5"
        with h5py.File(p, "w") as f:
            f["grid"] = np.full((4, 2, 3), cutoff, dtype=np.float32)
            f["sample_id"] = np.asarray([f"paired-{i}".encode() for i in range(4)])
            f["idx_train"] = np.asarray([0, 1])
            f["idx_val"] = np.asarray([2])
            f["idx_test"] = np.asarray([3])
            f["k_points"] = k
        targets[f"k{cutoff}_binary"] = {"dataset_path": str(p), "grid_field": "grid"}
    return source, targets


def test_cohort_audit_preserves_pairing_and_never_loads_labels(cohort) -> None:
    source, targets = cohort
    result = audit_transfer_cohort(source, targets)
    assert result["status"] == "paired_and_source_isolated"
    assert result["source_test_labels_loaded"] is False
    assert result["target_test_labels_loaded"] is False
    assert all(r["sample_id_overlap"] == r["input_grid_overlap"] == 0 for r in result["targets"].values())


@pytest.mark.parametrize("defect", ["id_overlap", "grid_overlap", "paired_split", "k_grid", "source_changed"])
def test_cohort_audit_rejects_leakage_and_identity_drift(cohort, defect: str) -> None:
    h5py = pytest.importorskip("h5py")
    source, targets = cohort
    target = targets["k8_binary"]["dataset_path"]
    if defect == "source_changed":
        with h5py.File(source / "source.h5", "r+") as f:
            f["grid"][0] = 99
    else:
        with h5py.File(target, "r+") as f:
            if defect == "id_overlap":
                del f["sample_id"]
                f["sample_id"] = np.asarray([b"paired-0", b"paired-1", b"paired-2", b"source-train"])
            elif defect == "grid_overlap":
                f["grid"][3] = 1.0
            elif defect == "paired_split":
                f["idx_train"][1] = 2
            else:
                f["k_points"][1] = [0.25, 0.0]
    with pytest.raises(RuntimeError):
        audit_transfer_cohort(source, targets)


def test_n56_worker_and_controller_preserve_resources_and_no_retry() -> None:
    worker = (ROOT / "jobs/paracloud_exp_fwd_023_evaluate.sh").read_text()
    controller = (ROOT / "jobs/paracloud_submit_exp_fwd_023.sh").read_text()
    assert "#SBATCH --partition=gpu_5090" in worker and "#SBATCH --gres=gpu:1" in worker
    assert "#SBATCH --cpus-per-task=8" in worker and "#SBATCH --mem" not in worker
    assert "%7" in controller and "start + 15" in controller and "no retry permitted" in controller
    assert "source six-domain evaluation must finalize first" in controller
    assert "run-training-task" not in worker + controller


def test_freeze_derivative_preserves_source_tasks_and_only_freezes_new_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from phc2d.forward import full_cell_transfer_contracts as module

    source = tmp_path / "source"
    source.mkdir()
    profile = yaml.safe_load((ROOT / "configs/research/full_cell_multidomain_exp_fwd_021_v1.yaml").read_text())
    (source / "config.resolved.yaml").write_text(yaml.safe_dump(profile))
    tasks = [{"task_id": f"{regime}_{model}_s0", "regime": regime, "model": model, "seed": 0, "run_dir": f"training/{regime}/{model}_s0"}
             for regime in profile["regimes"] for model in profile["models"]]
    checkpoints = [{**r, "checkpoint_sha256": f"unique-{i}"} for i, r in enumerate(tasks)]
    write_csv(source / "task_manifest.csv", tasks)
    write_csv(source / "evaluation/checkpoint_manifest.csv", checkpoints)
    write_csv(source / "normalizer_manifest.csv", [{"regime": "S1-B", "normalizer_path": "normalizers/source.npz"}])
    write_csv(source / "subset_manifest.csv", [{"regime": "S1-B", "domain": "k5_binary", "indices_path": "subsets/source.npy"}])
    write_csv(source / "dataset_manifest.csv", [{"domain": "k5_binary", "dataset_path": "source.h5"}])
    config = yaml.safe_load((ROOT / "configs/research/full_cell_transfer_exp_fwd_023_v1.yaml").read_text())
    config["source_campaign"] = str(source)
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    config["pairing_audit"] = str(provenance / "pairing_audit.json")
    Path(config["pairing_audit"]).write_text(json.dumps({"status": "passed"}))
    for domain, row in config["datasets"].items():
        (provenance / f"{domain}_hpc_audit.json").write_text(json.dumps({"status": "passed", "problems": [], "dataset_sha256": row["sha256"]}))
    config_path = tmp_path / "transfer.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    original_hashes = {p.name: sha256_file(p) for p in source.glob("*.csv")}
    monkeypatch.setattr(module, "audit_training_runs", lambda *a, **k: {"status": "training_slice_complete"})
    monkeypatch.setattr(module, "audit_dataset", lambda domain, row, path: {"domain": domain, "dataset_path": path, "dataset_sha256": row["sha256"], "grid_field": row["grid_field"]})
    monkeypatch.setattr(module, "audit_transfer_cohort", lambda *a: {"status": "paired_and_source_isolated"})
    def mock_freeze(campaign):
        inherited = yaml.safe_load((campaign / "config.resolved.yaml").read_text())
        assert inherited["experiment"]["experiment_id"] == "EXP-FWD-023"
        assert inherited["evaluation"]["targets"] == list(config["datasets"])
        assert inherited["training"] == profile["training"] and inherited["models"] == profile["models"]
        assert inherited["regimes"] == profile["regimes"]
        write_csv(campaign / "evaluation/checkpoint_manifest.csv", checkpoints)
        return campaign / "evaluation"
    monkeypatch.setattr(module, "freeze_evaluation", mock_freeze)
    child = module.freeze_transfer_evaluation(config_path, tmp_path / "child", repository_root=ROOT)
    inherited_tasks = read_csv(child / "task_manifest.csv")
    assert all(Path(row["run_dir"]).is_relative_to(source) for row in inherited_tasks)
    assert {p.name: sha256_file(p) for p in source.glob("*.csv")} == original_hashes
    assert checkpoint_identity_map(child) == checkpoint_identity_map(source)
    with pytest.raises(FileExistsError):
        module.freeze_transfer_evaluation(config_path, child, repository_root=ROOT)


def test_rendered_draft_contains_all_756_cells_and_boundaries(tmp_path: Path) -> None:
    from phc2d.reporting.full_cell_transfer import render_transfer_report

    profile = yaml.safe_load((ROOT / "configs/research/full_cell_multidomain_exp_fwd_021_v1.yaml").read_text())
    (tmp_path / "config.resolved.yaml").write_text(yaml.safe_dump(profile))
    combined = tmp_path / "evaluation/combined"
    combined.mkdir(parents=True)
    (combined / "results.json").write_text(json.dumps({"status": "finalized_twelve_domain_exploratory_evidence", "evaluation_cell_count": 756}))
    targets = [f"k{k}_{lane}" for k in (5, 8, 10, 15, 20) for lane in ("binary", "graded")] + ["f50_binary", "f50_graded"]
    checkpoints = [{"regime": regime, "model": model} for regime in profile["regimes"] for model in profile["models"]]
    rows = [{**cp, "target_domain": target, "band1_mre_pct": 3.0, "bands_2_10_mre_pct": 1.0, "bands_1_10_mre_pct": 1.2} for cp in checkpoints for target in targets]
    matrix, macros = merge_domain_matrix(rows, checkpoints, targets)
    write_csv(combined / "cell_matrix.csv", matrix)
    write_csv(combined / "twelve_domain_macro.csv", macros)
    write_csv(combined / "training_resources.csv", [{**cp, "training_status": "formal_training_success" if i < 23 else "training_complete_convergence_not_met"} for i, cp in enumerate(checkpoints)])
    output = render_transfer_report(tmp_path)
    text = output.read_text()
    assert "status: draft" in text and "WSL real-MPB 数据验收未完成" in text
    assert "23 个通过收敛门、40 个" in text and "跨 seed" in text
    assert text.count("1.0000 / 1.2000") == 756
    assert "训练来源→目标域" in text and "K15-B" in text and "F50-G" in text
    with pytest.raises(FileExistsError):
        render_transfer_report(tmp_path)


def test_summary_merges_two_halves_and_keeps_training_resources_unique(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from phc2d.forward import full_cell_transfer_summary as module

    source, child = tmp_path / "source", tmp_path / "child"
    checkpoints = [{"task_id": f"task-{i}", "regime": "S1-B", "model": model, "seed": 0, "checkpoint_sha256": f"sha-{i}"}
                   for i, model in enumerate(("mlp", "fno2d_band_v2"))]
    old_targets = ["k5_binary", "k5_graded", "k20_binary", "k20_graded", "f50_binary", "f50_graded"]
    new_targets = [f"k{k}_{lane}" for k in (8, 10, 15) for lane in ("binary", "graded")]
    profile = {"experiment": {"experiment_id": "EXP-FWD-021"}, "models": {r["model"]: {} for r in checkpoints},
               "regimes": {"S1-B": {"domains": {"k5_binary": 4000, "k20_binary": 4000}}},
               "datasets": {"k5_binary": {"target_shape": [2, 10]}}, "evaluation": {"targets": old_targets, "bootstrap_replicates": 4}}
    lookup = {}
    for parent, targets in ((source, old_targets), (child, new_targets)):
        parent.mkdir()
        evaluation = parent / "evaluation"
        evaluation.mkdir()
        write_csv(evaluation / "checkpoint_manifest.csv", checkpoints)
        write_csv(evaluation / "cache_manifest.csv", [{"target_domain": target, "cache_sha256": "toy"} for target in targets])
        (evaluation / "results.json").write_text(json.dumps({"status": "finalized_seed0_exploratory_surrogate_evidence"}))
        rows = [{**r, "target_domain": target, "transfer_group": "unseen_domain", "band1_mre_pct": 2, "bands_2_10_mre_pct": 1, "bands_1_10_mre_pct": 1.1}
                for r in checkpoints for target in targets]
        write_csv(evaluation / "cell_matrix.csv", rows)
        write_csv(evaluation / "paired_bootstrap.csv", [{"regime": "S1-B", "candidate_model": "mlp", "target_domain": target} for target in targets])
        dummy = evaluation / "dummy"
        dummy.mkdir()
        write_csv(dummy / "per_band_metrics.csv", [{"band_one_based": i, "mre_pct": 1} for i in range(1, 11)])
        np.savez(dummy / "sample_metrics.npz", sample_mre_bands_2_10_pct=np.asarray([1.0, 1.0]), sample_ids=np.asarray(["pair0", "pair1"]))
        lookup[parent] = [{**row, "cell_dir": "evaluation/dummy"} for row in rows]
        resolved = {**profile, "experiment": {"experiment_id": "EXP-FWD-021" if parent == source else "EXP-FWD-023"}, "evaluation": {**profile["evaluation"], "targets": targets}}
        if parent == child:
            resolved["transfer_evaluation"] = {"source_campaign": str(source), "source_checkpoint_manifest_sha256": sha256_file(source / "evaluation/checkpoint_manifest.csv"),
                                                "combined_cell_count": 24, "dataset_acceptance": "cluster_audited_pending_wsl_acceptance"}
        (parent / "config.resolved.yaml").write_text(yaml.safe_dump(resolved))
    tasks = []
    for checkpoint in checkpoints:
        run = source / f"training/{checkpoint['model']}"
        run.mkdir(parents=True)
        (run / "results.json").write_text(json.dumps({"status": "training_complete_convergence_not_met"}))
        (run / "metrics.json").write_text(json.dumps({"train_time_s": 10, "peak_gpu_memory_mb": 2}))
        tasks.append({**checkpoint, "run_dir": str(run)})
    write_csv(source / "task_manifest.csv", tasks)
    monkeypatch.setattr(module, "audit_training_runs", lambda *a: {"status": "training_slice_complete"})
    monkeypatch.setattr(module, "evaluation_status", lambda *a: {"status": "evaluation_complete"})
    monkeypatch.setattr(module, "evaluation_rows", lambda parent: lookup[parent])
    output = module.finalize_transfer_summary(child)
    result = json.loads((output / "results.json").read_text())
    assert result["evaluation_cell_count"] == 24 and result["per_band_record_count"] == 240
    assert len(read_csv(output / "training_resources.csv")) == 2
    assert json.loads((output / "metrics.json").read_text())["training_time_s_total_unique"] == 20
    assert len(read_csv(output / "paired_k_shift_bootstrap.csv")) == 8
    assert result["physical_claim"] is False and result["sample_bootstrap_is_not_cross_seed_stability"] is True
    with pytest.raises(FileExistsError):
        module.finalize_transfer_summary(child)
