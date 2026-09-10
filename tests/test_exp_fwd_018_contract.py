"""Frozen configuration and N40 job tests for EXP-FWD-018.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_exp_fwd_018_configs_preserve_the_paired_full_split_contract() -> None:
    baseline = yaml.safe_load((ROOT / "configs/research/full_cell_k5_frequency_weighted_exp_fwd_018_baseline_v1.yaml").read_text(encoding="utf-8"))
    weighted = yaml.safe_load((ROOT / "configs/research/full_cell_k5_frequency_weighted_exp_fwd_018_weighted_v1.yaml").read_text(encoding="utf-8"))
    for config in (baseline, weighted):
        assert config["experiment"]["experiment_id"] == "EXP-FWD-018"
        assert config["experiment"]["held_out_test_authorized"] is False
        assert config["training"]["train_size"] == 8000
        assert config["training"]["max_epochs"] == 200
        assert config["training"]["lane_batch_size"] == {"binary": 32, "graded": 32}
        assert config["training"]["checkpoint_selection"] == "equal_lane_validation_bands_2_10_mre"
        assert config["runtime"]["amp_enabled"] is False and config["runtime"]["tf32_enabled"] is False
    loss = weighted["training"]["loss"]
    assert loss["epsilon_f"] == 0.0008457602483520898
    assert loss["rho"] == 4.0 and loss["alpha"] == 0.25
    assert loss["center_kind"] == "heuristic_folded_center_x"


def test_exp_fwd_018_submitter_is_validation_only_and_bounded_to_three_gpus() -> None:
    submitter = (ROOT / "jobs/paracloud_submit_exp_fwd_018_frequency_weighted.sh").read_text(encoding="utf-8")
    worker = (ROOT / "jobs/paracloud_exp_fwd_018_frequency_weighted_train.sh").read_text(encoding="utf-8")
    assert "PHC2D_AUTHORIZE_EXP_FWD_018" in submitter
    assert "--array=0-5%3" in submitter
    assert "validation_only_no_test" in submitter
    assert "configs.frozen.sha256" in submitter
    assert "no automatic retry" in submitter
    assert worker.count('arm="baseline"') == 3
    assert worker.count('arm="weighted"') == 3
    assert "--head-kind dense" in worker
