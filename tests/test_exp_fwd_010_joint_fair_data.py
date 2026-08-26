"""Contracts for EXP-FWD-010 fair-total-data joint training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from phc2d.forward import full_cell_dual_lane
from phc2d.forward.full_cell_joint_fair_evaluation import freeze_evaluation


ROOT = Path(__file__).resolve().parents[1]


def test_exp_fwd_010_joint_uses_four_thousand_samples_per_lane() -> None:
    path = ROOT / "configs/research/full_cell_k5_forward_010_dual_lane_4k_seed0_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["experiment"]["experiment_id"] == "EXP-FWD-010"
    assert config["training"]["train_size"] == 4_000
    assert config["training"]["train_subset_salt"] == "exp-fwd-010-k5-joint-fair-4k-v1"
    assert config["training"]["lane_batch_size"] == {"binary": 32, "graded": 32}
    assert config["training"]["required_stop_epoch"] == 200
    assert "continuation" not in config
    resolved = full_cell_dual_lane._validate_and_resolve(
        config,
        seed=0,
        admission_probe=False,
        head_kind="dense",
    )
    assert resolved["training"]["train_size"] == 4_000
    assert resolved["training"]["max_epochs"] == 200


def test_exp_fwd_010_submitter_is_explicit_and_single_task() -> None:
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_010_joint_4k.sh").read_text(encoding="utf-8")
    worker = (ROOT / "jobs/paracloud_exp_fwd_010_joint_4k_train.sh").read_text(encoding="utf-8")
    assert "PHC2D_AUTHORIZE_EXP_FWD_010" in submitter
    assert "no automatic retry" in submitter
    assert "--array" not in submitter
    assert "full_cell_k5_forward_010_dual_lane_4k_seed0_v1.yaml" in worker
    assert "--seed 0 --head-kind dense" in worker


def test_exp_fwd_010_evaluation_contract_is_incremental_six_domain() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_k5_joint_4k_six_domain_exp_fwd_010_n40_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["experiment_id"] == "EXP-FWD-010"
    assert config["domains"] == ["K5-B", "K5-G", "K20-B", "K20-G", "F50-B", "F50-G"]
    assert config["baseline_identity"] == {
        "source_family": "K5",
        "source_model": "dual_lane_band_operator_v1",
        "source_lane": "J",
        "source_seed": 0,
    }
    assert config["statistics"]["paired_bootstrap_replicates"] == 10_000


def test_exp_fwd_010_freeze_reuses_baseline_cache_and_plans_only_six_cells(tmp_path: Path) -> None:
    domains = ["K5-B", "K5-G", "K20-B", "K20-G", "F50-B", "F50-G"]
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "results.json").write_text(
        json.dumps({"status": "evaluation_complete_pending_report"}), encoding="utf-8"
    )
    (baseline / "config.resolved.yaml").write_text("datasets: {frozen: true}\n", encoding="utf-8")
    with (baseline / "evaluation_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_family",
                "source_model",
                "source_lane",
                "source_seed",
                "target_domain",
                "evaluation_id",
                "shard_id",
            ],
        )
        writer.writeheader()
        for index, domain in enumerate(domains):
            writer.writerow(
                {
                    "source_family": "K5",
                    "source_model": "dual_lane_band_operator_v1",
                    "source_lane": "J",
                    "source_seed": 0,
                    "target_domain": domain,
                    "evaluation_id": f"baseline_{index}",
                    "shard_id": index,
                }
            )
    with (baseline / "test_cache_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "cache_path"])
        writer.writeheader()
        for domain in domains:
            writer.writerow({"domain": domain, "cache_path": f"cache/{domain}.h5"})
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "EXP-FWD-010",
                "domains": domains,
                "baseline_campaign": str(baseline),
                "baseline_identity": {
                    "source_family": "K5",
                    "source_model": "dual_lane_band_operator_v1",
                    "source_lane": "J",
                    "source_seed": 0,
                },
                "statistics": {"paired_bootstrap_replicates": 10_000},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    campaign = freeze_evaluation(config, checkpoint, tmp_path / "evaluation", repository_root=tmp_path)
    with (campaign / "evaluation_plan.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert [row["target_domain"] for row in rows] == domains
    assert {row["shard_id"] for row in rows} == {"0"}
    assert all(row["checkpoint_sha256"] == rows[0]["checkpoint_sha256"] for row in rows)
