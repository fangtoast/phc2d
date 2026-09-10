"""Contract tests for EXP-FWD-016 ranked-gap short screening.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from phc2d.forward.dual_lane_contracts import validate_and_resolve_dual_lane
from phc2d.forward.gap_aware_loss import RANKED_GAP_LOSS_NAME


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/full_cell_k5_gap_ranked_exp_fwd_016_seed0_v2.yaml"


def test_exp_fwd_016_is_one_fair_validation_only_ranked_gap_arm() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolved = validate_and_resolve_dual_lane(
        raw, seed=0, admission_probe=False, head_kind="dense"
    )
    assert resolved["experiment"]["experiment_id"] == "EXP-FWD-016"
    assert resolved["experiment"]["held_out_test_authorized"] is False
    assert resolved["training"]["train_size"] == 4000
    assert resolved["training"]["max_epochs"] == 20
    assert resolved["training"]["checkpoint_selection"] == (
        "equal_lane_validation_bands_2_10_mre"
    )
    loss = resolved["training"]["loss"]
    assert loss["name"] == RANKED_GAP_LOSS_NAME
    assert loss["candidate_ranking_weight"] == 0.05
    assert loss["complete_gap_recall_weight"] == 0.10


def test_exp_fwd_016_submitter_requires_explicit_authorization_and_has_no_retry() -> None:
    submitter = (
        ROOT / "jobs/paracloud_submit_exp_fwd_016_gap_ranked.sh"
    ).read_text(encoding="utf-8")
    worker = (ROOT / "jobs/paracloud_exp_fwd_016_gap_ranked_train.sh").read_text(
        encoding="utf-8"
    )
    assert "PHC2D_AUTHORIZE_EXP_FWD_016" in submitter
    assert "explicit authorization id is required" in submitter
    assert "no automatic retry" in submitter
    assert "--array" not in submitter
    assert "full_cell_k5_gap_ranked_exp_fwd_016_seed0_v2.yaml" in submitter
    assert "python -m phc2d.forward.full_cell_dual_lane" in worker
