"""Contract tests for EXP-FWD-007 K5 training and six-domain evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from phc2d.forward import full_cell_six_domain_n40 as six
from phc2d.forward import full_cell_training_campaign as training


ROOT = Path(__file__).resolve().parents[1]


def _audit(lane: str) -> dict[str, object]:
    return {
        "lane": lane, "dataset_sha256": lane * 64, "idx_train_hash": "t" * 64,
        "idx_val_hash": "v" * 64, "idx_test_hash": "e" * 64, "k_points_sha256": "k" * 64,
    }


def test_k5_configs_freeze_hashes_and_training_contract() -> None:
    binary = yaml.safe_load((ROOT / "configs/research/full_cell_k5_forward_binary_seed0_v1.yaml").read_text(encoding="utf-8"))
    graded = yaml.safe_load((ROOT / "configs/research/full_cell_k5_forward_graded_seed0_v1.yaml").read_text(encoding="utf-8"))
    assert binary["dataset"]["expected_dataset_sha256"] == "7d2c6bef128c7ee61e0c5ca0eb9561b95992ccf8e53f325274c4889752ea86b6"
    assert graded["dataset"]["expected_dataset_sha256"] == "eb140534d941cbdb53addca279f00337810124a34fade20a19bfb969e0e6af20"
    for config in (binary, graded):
        assert config["training"]["max_epochs"] == 100
        assert config["training"]["early_stopping_patience"] == 15
        assert config["training"]["micro_batch_size"] == 64
        assert config["runtime"]["amp_enabled"] is False and config["runtime"]["tf32_enabled"] is False
        assert set(config["models"]) == set(six.MODELS)


def test_k5_profile_has_exactly_seventeen_tasks_and_three_waves(tmp_path: Path) -> None:
    profile = yaml.safe_load((ROOT / "configs/research/exp_fwd_007_k5_campaign_profile_v1.yaml").read_text(encoding="utf-8"))
    rows = training._task_rows(tmp_path, profile, {"B": _audit("B"), "G": _audit("G")}, stage="FORMAL")
    assert len(rows) == 17
    assert [row["wave"] for row in rows].count("W1") == 8
    assert [row["wave"] for row in rows].count("W2") == 8
    assert [row["wave"] for row in rows].count("W3") == 1
    assert {row["seed"] for row in rows} == {0}
    assert rows[-1]["train_lane"] == "J"
    assert training._admission_submission_units(profile, rows) == (15, 2)


def test_k5_admission_submission_units_preserve_all_seventeen_tasks(tmp_path: Path) -> None:
    profile = yaml.safe_load((ROOT / "configs/research/exp_fwd_007_k5_campaign_profile_v1.yaml").read_text(encoding="utf-8"))
    rows = training._task_rows(tmp_path, profile, {"B": _audit("B"), "G": _audit("G")}, stage="ADMISSION")
    manifests = training._write_admission_submission_manifests(tmp_path, profile, rows)
    first = training._read_csv(manifests[0]); second = training._read_csv(manifests[1])
    assert [len(first), len(second)] == [15, 2]
    assert {int(row["source_task_index"]) for row in first + second} == set(range(17))
    assert [int(row["task_index"]) for row in first] == list(range(15))
    assert [int(row["task_index"]) for row in second] == list(range(2))


def test_admission_b_requires_passing_a_audit_and_preserves_test_boundary(tmp_path: Path) -> None:
    profile = yaml.safe_load((ROOT / "configs/research/exp_fwd_007_k5_campaign_profile_v1.yaml").read_text(encoding="utf-8"))
    rows = training._task_rows(tmp_path, profile, {"B": _audit("B"), "G": _audit("G")}, stage="ADMISSION")
    manifests = training._write_admission_submission_manifests(tmp_path, profile, rows)
    for manifest in manifests:
        for row in training._read_csv(manifest):
            run = Path(row["run_dir"]); run.mkdir(parents=True)
            (run / "model.pt").write_bytes(b"admission")
            (run / "results.json").write_text(json.dumps({
                "status": "dual_lane_admission_probe_pass" if row["train_lane"] == "J" else "full_cell_forward_admission_probe_pass",
                "formal_training_started": False, "test_executed": False,
            }), encoding="utf-8")
            (run / "metrics.json").write_text(json.dumps({"loss": 0.25, "peak_gpu_memory_mb": 1.0}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="before a passing admission A audit"):
        training.audit_admission(tmp_path, phase="B")
    assert training.audit_admission(tmp_path, phase="A").is_file()
    audit_b = json.loads(training.audit_admission(tmp_path, phase="B").read_text(encoding="utf-8"))
    assert audit_b["status"] == "pass"
    assert audit_b["task_count"] == 2
    assert audit_b["test_read_before_checkpoint_freeze"] is False


def _checkpoints() -> list[dict[str, object]]:
    rows = []
    for family, seeds in (("K5", (0,)), ("K20", (0,)), ("F50", (0, 1, 2))):
        for seed in seeds:
            for model in six.MODELS:
                for lane in ("B", "G"):
                    rows.append({"source_family": family, "model": model, "source_lane": lane, "source_seed": seed, "checkpoint_path": f"{family}_{model}_{lane}_{seed}.pt", "checkpoint_sha256": f"{len(rows):064x}"})
            rows.append({"source_family": family, "model": six.JOINT_MODEL, "source_lane": "J", "source_seed": seed, "checkpoint_path": f"{family}_joint_{seed}.pt", "checkpoint_sha256": f"{len(rows):064x}"})
    return rows


def test_six_domain_exact_85_510_10_and_54_preflight_contract() -> None:
    checkpoints = _checkpoints(); cells = six.build_evaluation_cells(checkpoints)
    assert len(checkpoints) == six.EXPECTED_CHECKPOINTS == 85
    assert len(cells) == six.EXPECTED_CELLS == 510
    assert {row["target_domain"] for row in cells} == set(six.DOMAINS)
    assert all(sum(row["target_domain"] == domain for row in cells) == 85 for domain in six.DOMAINS)
    assert {int(row["shard_id"]) for row in cells} == set(range(six.EXPECTED_SHARDS))
    identities = {(r["source_family"], r["model"], r["source_lane"], target) for r in checkpoints for target in (("B", "G") if r["source_lane"] == "J" else (r["source_lane"],))}
    assert len(identities) == six.EXPECTED_PREFLIGHTS == 54


def test_exp_fwd_007_jobs_are_authorized_and_read_only_status() -> None:
    submit = (ROOT / "jobs/paracloud_submit_exp_fwd_007_k5.sh").read_text(encoding="utf-8")
    evaluation = (ROOT / "jobs/paracloud_submit_exp_fwd_007_six_domain.sh").read_text(encoding="utf-8")
    status = (ROOT / "jobs/paracloud_exp_fwd_007_status.sh").read_text(encoding="utf-8")
    assert "PHC2D_AUTHORIZE_EXP_FWD_007" in submit and "--array=0-$((count-1))%8" in submit
    assert "ADMISSION_A|ADMISSION_B|W1|W2|W3" in submit
    assert "admission B requires a passing admission A audit" in submit
    assert "--array=0-9%8" in evaluation and "--dependency=\"afterok:$job_id\"" in evaluation
    assert "sbatch" not in status and "squeue" in status and "sacct" in status
    assert "--campaign-dir" in status and "admission_a_manifest.csv" in status and "admission_b_manifest.csv" in status
    assert 'sacct -j "$job_ids" --format' in status
