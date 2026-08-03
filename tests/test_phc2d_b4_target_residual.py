"""Contract tests for the validation-only B4 target-residual diagnostic.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from phc2d.forward.field_eigen_b4 import is_field_eigen_b4_dense_config
from phc2d.forward.field_eigen_b4_target_residual import (
    _calibration_rows,
    _candidate_decision,
    _pilot_candidate_decision,
    _structure_scores,
    _unique_k_indices,
    _validate_contract,
    is_field_eigen_b4_target_residual_config,
)


def _raw_row(
    *,
    source: str,
    target_residual: float,
    self_residual: float,
    frequency_mre: float,
) -> dict[str, object]:
    return {
        "case": "P4_seed0",
        "candidate": "P4",
        "seed": 0,
        "source_sample_id": source,
        "source_sample_hash": f"hash-{source}",
        "target_residual_score_component": math.log1p(target_residual**2),
        "self_relative_residual": self_residual,
        "frequency_mre_pct": frequency_mre,
        "rayleigh_target_log_error": 0.1,
    }


def test_structure_score_uses_median_log1p_target_residual_squared() -> None:
    rows = [
        _raw_row(source="sample_1", target_residual=1.0, self_residual=2.0, frequency_mre=3.0),
        _raw_row(source="sample_1", target_residual=3.0, self_residual=4.0, frequency_mre=5.0),
        _raw_row(source="sample_1", target_residual=2.0, self_residual=3.0, frequency_mre=4.0),
    ]
    score = _structure_scores(rows)[0]
    assert score["target_residual_score"] == pytest.approx(math.log1p(2.0**2))
    assert score["frequency_mre_pct"] == pytest.approx(4.0)
    assert score["condition_count"] == 3
    assert score["all_finite"] is True


def test_unique_k_indices_drops_only_declared_duplicate_coordinates() -> None:
    sidecar = {
        "k_points": [[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]],
        "duplicate_coordinate_groups": [[0, 2]],
    }
    assert _unique_k_indices(sidecar) == [0, 1]


def test_target_residual_calibration_uses_structure_level_bootstrap() -> None:
    rows = [
        {
            "case": "P4_seed0",
            "candidate": "P4",
            "seed": 0,
            "source_sample_id": f"sample_{index}",
            "target_residual_score": float(index),
            "frequency_mre_pct": float(index),
            "all_finite": True,
        }
        for index in range(1, 17)
    ]
    calibration = _calibration_rows(
        rows,
        bootstrap_seed=7,
        bootstrap_repeats=100,
        thresholds={
            "spearman_ci_lower_min_exclusive": 0.0,
            "quartile_frequency_mre_reduction_min_pct": 10.0,
        },
    )[0]
    assert calibration["structure_count"] == 16
    assert calibration["spearman_bootstrap_ci_lower"] > 0.0
    assert calibration["quartile_frequency_mre_reduction_pct"] > 10.0
    assert calibration["physics_target_residual_64_ready"] == "pass"


def test_fixed_resolution_selection_does_not_require_cross_resolution_pass() -> None:
    candidates = {
        "P4": [
            {"name": f"P4_seed{seed}", "seed": seed}
            for seed in (0, 1, 2)
        ],
        "P5": [
            {"name": f"P5_seed{seed}", "seed": seed}
            for seed in (0, 1, 2)
        ],
    }
    calibration = [
        {
            "case": f"{candidate}_seed{seed}",
            "candidate": candidate,
            "seed": seed,
            "physics_target_residual_64_ready": "pass" if candidate == "P4" else "blocked",
        }
        for candidate in ("P4", "P5")
        for seed in (0, 1, 2)
    ]
    structure_rows = [
        {
            "case": f"{candidate}_seed{seed}",
            "candidate": candidate,
            "seed": seed,
            "frequency_mre_pct": 2.0 if candidate == "P4" else 1.0,
        }
        for candidate in ("P4", "P5")
        for seed in (0, 1, 2)
    ]
    source = {
        "candidate_decisions": {
            candidate: {
                "physics_frequency_crosscheck_ready": {"status": "pass"},
                "cross_resolution_zero_shot_pass": {"status": "blocked"},
            }
            for candidate in ("P4", "P5")
        }
    }
    decision = _candidate_decision(
        calibration,
        structure_rows,
        candidates=candidates,
        source_decision=source,
        thresholds={
            "spearman_ci_lower_min_exclusive": 0.0,
            "quartile_frequency_mre_reduction_min_pct": 10.0,
        },
        required_seeds={0, 1, 2},
    )
    assert decision["selected_candidate"] == "P4"
    assert decision["physics_target_residual_64_ready"]["status"] == "pass"
    assert decision["pino_inverse_rerank_64_ready"]["status"] == "blocked"
    assert decision["pino_inverse_rerank_multires_ready"]["status"] == "blocked"
    assert decision["next_required_stage"] == "pino_inverse_adapter_preflight_64"


def test_config_is_validation_only_and_pins_six_checkpoints() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "research" / "field_eigen_te_b4_target_residual_64.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert is_field_eigen_b4_target_residual_config(path)
    assert is_field_eigen_b4_dense_config(path)
    assert _validate_contract(raw) == ("val", {0, 1, 2}, 64)
    assert raw["validation_contract"]["allow_test"] is False
    assert raw["evaluation"]["unique_coordinate_count"] == 30
    assert set(raw["candidates"]) == {"P4", "P5"}
    assert sum(len(value["cases"]) for value in raw["candidates"].values()) == 6
    assert all(
        len(case["checkpoint_sha256"]) == 64
        for candidate in raw["candidates"].values()
        for case in candidate["cases"]
    )


def test_contract_rejects_test_access() -> None:
    raw = {
        "validation_contract": {
            "evaluation_split": "test",
            "allow_test": True,
            "required_seeds": [0, 1, 2],
        },
        "evaluation": {"resolution": 64},
    }
    with pytest.raises(ValueError, match="validation-only"):
        _validate_contract(raw)


def test_seed0_pilot_contract_is_isolated_from_formal_readiness() -> None:
    raw = {
        "experiment": {
            "execution": "field_eigen_te_p6_seed0_target_residual_pilot",
        },
        "validation_contract": {
            "evaluation_split": "val",
            "allow_test": False,
            "required_seeds": [0],
            "pilot_only": True,
            "single_seed_inverse_admission_allowed": True,
            "formal_three_seed_claim_allowed": False,
        },
        "evaluation": {"resolution": 64},
    }
    assert _validate_contract(raw) == ("val", {0}, 64)

    raw["validation_contract"]["formal_three_seed_claim_allowed"] = True
    with pytest.raises(ValueError, match="forbid a formal three-seed claim"):
        _validate_contract(raw)


def test_seed0_pilot_pass_allows_conditional_adapter_admission_only() -> None:
    candidates = {
        "P6A": [{"name": "P6A_target_pde_seed0", "seed": 0}],
        "P6B": [{"name": "P6B_target_pde_rayleigh_seed0", "seed": 0}],
    }
    calibration = [
        {
            "case": "P6A_target_pde_seed0",
            "candidate": "P6A",
            "seed": 0,
            "physics_target_residual_64_ready": "pass",
        },
        {
            "case": "P6B_target_pde_rayleigh_seed0",
            "candidate": "P6B",
            "seed": 0,
            "physics_target_residual_64_ready": "blocked",
        },
    ]
    structure_rows = [
        {
            "case": case["name"],
            "candidate": candidate,
            "seed": 0,
            "frequency_mre_pct": 2.0 if candidate == "P6A" else 1.0,
        }
        for candidate, cases in candidates.items()
        for case in cases
    ]
    decision = _pilot_candidate_decision(
        calibration,
        structure_rows,
        candidates=candidates,
        thresholds={
            "spearman_ci_lower_min_exclusive": 0.0,
            "quartile_frequency_mre_reduction_min_pct": 10.0,
        },
    )
    assert decision["status"] == "conditional_inverse_admission"
    assert decision["selected_candidate"] == "P6A"
    assert (
        decision["candidate_decisions"]["P6A"]["physics_target_residual_64_ready"]["status"]
        == "pass"
    )
    assert (
        decision["candidate_decisions"]["P6B"]["physics_target_residual_64_ready"]["status"]
        == "blocked"
    )
    assert decision["physics_target_residual_64_ready"]["status"] == "pass"
    assert decision["pino_inverse_adapter_preflight_ready"]["status"] == "pass"
    assert decision["pino_inverse_rerank_64_ready"]["status"] == "blocked"


def test_p6_seed0_pilot_config_freezes_two_checkpoints_and_rejects_test() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "configs"
        / "research"
        / "field_eigen_te_p6_seed0_target_residual_pilot_64.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert is_field_eigen_b4_target_residual_config(path)
    assert is_field_eigen_b4_dense_config(path)
    assert _validate_contract(raw) == ("val", {0}, 64)
    assert raw["validation_contract"]["allow_test"] is False
    assert raw["validation_contract"]["single_seed_inverse_admission_allowed"] is True
    assert raw["validation_contract"]["formal_three_seed_claim_allowed"] is False
    assert set(raw["candidates"]) == {"P6A", "P6B"}
    cases = [
        case
        for candidate in raw["candidates"].values()
        for case in candidate["cases"]
    ]
    assert len(cases) == 2
    assert all(case["seed"] == 0 for case in cases)
    assert all(len(case["checkpoint_sha256"]) == 64 for case in cases)
