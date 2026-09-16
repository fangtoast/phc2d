"""Contract tests for EXP-FWD-022 branch attribution.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np
import yaml

from phc2d.forward.global_local_branch_attribution import (
    INTERVENTIONS,
    diagnostic_indices,
    output_distance,
)


def test_exp_fwd_022_config_freezes_validation_matrix() -> None:
    config = yaml.safe_load(
        open(
            "configs/research/full_cell_global_local_branch_attribution_exp_fwd_022_v1.yaml",
            encoding="utf-8",
        )
    )
    assert config["experiment"]["experiment_id"] == "EXP-FWD-022"
    assert len(config["datasets"]) == 6
    assert config["matrix"]["checkpoint_count"] == 36
    assert config["matrix"]["intervention_cells"] == 1080
    assert config["runtime"]["target_platform"] == "paracloud-n40"
    assert config["authorization"]["prohibited"] == [
        "automatic_retry",
        "automatic_independent_training",
        "test_driven_change",
        "model_promotion",
        "real_MPB",
        "real_Meep",
        "physical_claim",
    ]


def test_intervention_identity_is_complete_and_unique() -> None:
    assert INTERVENTIONS == (
        "full",
        "g_off_raw",
        "l_off_raw",
        "g_off_renorm",
        "l_off_renorm",
    )
    assert len(set(INTERVENTIONS)) == 5


def test_output_distance_uses_prediction_change_without_target() -> None:
    baseline = np.ones((2, 3, 10), dtype=np.float32)
    candidate = baseline.copy()
    candidate[:, :, 1:10] *= 2.0
    observed = output_distance(candidate, baseline)
    assert np.allclose(observed, 1.0)


def test_diagnostic_indices_are_reproducible_and_in_range() -> None:
    rng = np.random.default_rng(42)
    values = rng.random((1000, 1, 8, 8), dtype=np.float32)
    first = diagnostic_indices(values, count=512, key="checkpoint:target")
    second = diagnostic_indices(values, count=512, key="checkpoint:target")
    assert set(first) == {"random", "structure_stratified"}
    for name in first:
        assert len(first[name]) == 512
        assert len(np.unique(first[name])) == 512
        assert np.array_equal(first[name], second[name])
        assert int(first[name].min()) >= 0
        assert int(first[name].max()) < len(values)
