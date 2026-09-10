"""Contracts for EXP-FWD-011 fair-joint continuation to total epoch 300.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phc2d.forward import full_cell_dual_lane


ROOT = Path(__file__).resolve().parents[1]


def test_exp_fwd_011_is_fixed_epoch300_weight_only_continuation() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_k5_forward_011_dual_lane_4k_epoch300_seed0_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["experiment"]["experiment_id"] == "EXP-FWD-011"
    assert config["training"]["train_size"] == 4_000
    assert config["training"]["train_subset_salt"] == "exp-fwd-010-k5-joint-fair-4k-v1"
    assert config["training"]["max_epochs"] == 300
    assert "required_stop_epoch" not in config["training"]
    assert config["continuation"] == {
        **config["continuation"],
        "mode": "epoch200_weights_optimizer_reset",
        "source_experiment": "EXP-FWD-010",
        "source_epoch": 200,
        "optimizer_reset": True,
        "data_order_state_restored": False,
        "check_interval": 100,
        "relative_threshold": 0.005,
        "max_total_epoch": 300,
        "stop_when_converged": False,
    }
    resolved = full_cell_dual_lane._validate_and_resolve(
        config, seed=0, admission_probe=False, head_kind="dense"
    )
    assert resolved["training"]["max_epochs"] == 300


def test_exp_fwd_011_rejects_early_convergence_stop() -> None:
    path = ROOT / "configs/research/full_cell_k5_forward_011_dual_lane_4k_epoch300_seed0_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["continuation"]["stop_when_converged"] = True
    with pytest.raises(ValueError, match="run through total epoch 300"):
        full_cell_dual_lane._validate_and_resolve(
            config, seed=0, admission_probe=False, head_kind="dense"
        )


def test_exp_fwd_011_evaluation_and_submit_contracts() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_k5_joint_4k_epoch300_six_domain_exp_fwd_011_n40_v1.yaml").read_text(encoding="utf-8")
    )
    assert config["experiment_id"] == "EXP-FWD-011"
    assert config["baseline_label"] == "EXP-FWD-010 fair 4k+4k epoch-200 joint"
    assert config["comparison_output"] == "comparison_to_exp_fwd_010_epoch200_joint.csv"
    assert config["statistics"]["paired_bootstrap_replicates"] == 10_000
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_011_joint_4k_epoch300.sh").read_text(encoding="utf-8")
    worker = (ROOT / "jobs/paracloud_exp_fwd_011_joint_4k_epoch300_train.sh").read_text(encoding="utf-8")
    eval_submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_011_joint_4k_epoch300_eval.sh").read_text(encoding="utf-8")
    assert "PHC2D_AUTHORIZE_EXP_FWD_011" in submitter
    assert "no automatic retry" in submitter
    assert "--array" not in submitter
    assert "--warm-start-run" in worker
    assert "full_cell_k5_forward_011_dual_lane_4k_epoch300_seed0_v1.yaml" in worker
    assert '"$PHC2D_ENV/bin/python"' in eval_submitter
