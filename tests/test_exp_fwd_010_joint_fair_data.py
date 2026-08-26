"""Contracts for EXP-FWD-010 fair-total-data joint training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from phc2d.forward import full_cell_dual_lane


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
