"""Contract tests for the paired EXP-FWD-014 gap-decision screen.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from phc2d.forward.dual_lane_contracts import validate_and_resolve_dual_lane
from phc2d.forward.gap_aware_loss import GAP_AWARE_LOSS_NAME


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "research"
BASELINE = CONFIGS / "full_cell_k5_gap_screen_exp_fwd_014_baseline_seed0_v1.yaml"
GAP_AWARE = CONFIGS / "full_cell_k5_gap_screen_exp_fwd_014_gap_aware_seed0_v1.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_exp_fwd_014_arms_share_the_fair_short_budget_contract() -> None:
    baseline = _load(BASELINE)
    gap_aware = _load(GAP_AWARE)
    for payload in (baseline, gap_aware):
        resolved = validate_and_resolve_dual_lane(
            payload, seed=0, admission_probe=False, head_kind="dense"
        )
        assert resolved["experiment"]["experiment_id"] == "EXP-FWD-014"
        assert resolved["training"]["train_size"] == 4000
        assert resolved["training"]["max_epochs"] == 20
        assert resolved["training"]["early_stopping_patience"] == 0
        assert resolved["evaluation"]["gap_decision"] == {
            "lower_band_min_one_based": 2,
            "lower_band_max_one_based": 9,
        }
    for key in ("datasets", "representation", "model", "runtime", "evaluation"):
        assert baseline[key] == gap_aware[key]
    for key in (
        "train_size",
        "train_subset_salt",
        "max_epochs",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "micro_batch_size",
        "effective_batch_size",
        "lane_batch_size",
        "checkpoint_selection",
    ):
        assert baseline["training"][key] == gap_aware["training"][key]


def test_exp_fwd_014_changes_only_the_registered_gap_objective() -> None:
    baseline = _load(BASELINE)
    gap_aware = _load(GAP_AWARE)
    assert baseline["training"]["loss"] == "equal_lane_normalized_full_matrix_mse"
    loss = gap_aware["training"]["loss"]
    assert loss["name"] == GAP_AWARE_LOSS_NAME
    assert loss["selected_upper_profile_weight"] == 0.10
    assert gap_aware["experiment"]["arm"] == "gap_aware_with_upper_profile"
    assert baseline["experiment"]["arm"] == "baseline_mse"
