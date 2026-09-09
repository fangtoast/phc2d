"""Contract tests for EXP-FWD-020 global-local bridge.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from pathlib import Path

import pytest
import yaml

from phc2d.models import build_model, get_model_spec


ROOT = Path(__file__).resolve().parents[1]


def test_exp_fwd_020_is_registered_but_remote_training_is_not_authorized() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/research/full_cell_global_local_bridge_exp_fwd_020_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["experiment"] == {
        "experiment_id": "EXP-FWD-020",
        "family": "full_cell_global_local_bridge",
        "evidence_status": "implemented_not_authorized",
        "primary_layer": "L2_method_architecture",
    }
    assert config["split"]["test_targets_loaded"] is False
    assert config["model"]["paired_variants"] == {
        "baseline": {"local_kernel": 3},
        "candidate": {"local_kernel": 7},
    }
    assert config["authorization"]["granted"] is False
    assert "remote_submission" in config["authorization"]["prohibited_until_granted"]

    spec = get_model_spec("global_local_band_operator_v1")
    assert spec.accepts_k_points is True
    assert spec.accepts_lane_id is False
    assert spec.defaults["local_kernel"] == 7


def test_exp_fwd_020_kernel_pair_stays_within_frozen_capacity_gate() -> None:
    pytest.importorskip("torch")
    baseline = build_model("global_local_band_operator_v1", local_kernel=3)
    candidate = build_model("global_local_band_operator_v1", local_kernel=7)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    assert baseline_count == 2_571_978
    assert candidate_count == 2_587_338
    assert candidate_count / baseline_count <= 1.05
