"""Contract tests for validation-only B4 repaired-candidate selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from phc2d.forward.field_eigen_b4_selector import _candidate_cases, _select_candidate


def _decision(status: str) -> dict[str, object]:
    return {
        "physics_frequency_crosscheck_ready": {"status": status},
        "physics_residual_rerank_ready": {"status": status},
        "cross_resolution_zero_shot_pass": {"status": status},
    }


def test_selector_tie_break_prefers_worst_seed_frequency_then_128_residual() -> None:
    decisions = {"P4": _decision("pass"), "P5": _decision("pass")}
    selected = _select_candidate(decisions, {"P4": (1.01, 12.0), "P5": (1.01, 8.0)})
    assert selected["status"] == "selected"
    assert selected["selected_candidate"] == "P5"
    assert selected["inverse_status"] == "blocked"


def test_selector_blocks_when_any_candidate_gate_fails_or_score_is_nonfinite() -> None:
    selected = _select_candidate({"P4": _decision("blocked"), "P5": _decision("blocked")}, {"P4": (float("inf"), float("inf")), "P5": (float("inf"), float("inf"))})
    assert selected["status"] == "blocked"
    assert selected["selected_candidate"] is None


def test_selector_requires_exact_three_candidate_seed_cases() -> None:
    raw = {"candidate_model": {"name": "fno_field_v2", "params": {"width": 4}}, "candidates": {"P4": {"cases": [{"name": "P4_seed0", "seed": 0, "checkpoint_path": "a.pt", "checkpoint_sha256": "a" * 64}, {"name": "P4_seed1", "seed": 1, "checkpoint_path": "b.pt", "checkpoint_sha256": "b" * 64}]}}}
    with pytest.raises(ValueError, match="exactly seeds"):
        _candidate_cases(raw, required_seeds={0, 1, 2})


def test_validation_selector_config_is_test_isolated_and_pins_six_checkpoints() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "research" / "field_eigen_te_b4_repair_validation_selector.yaml").read_text(encoding="utf-8"))
    assert raw["validation_contract"] == {"evaluation_split": "val", "allow_test": False, "required_seeds": [0, 1, 2]}
    assert raw["dense_k"]["structures_per_split"] == 32
    assert set(raw["candidates"]) == {"P4", "P5"}
    assert all(len(candidate["cases"]) == 3 for candidate in raw["candidates"].values())
    assert all(len(case["checkpoint_sha256"]) == 64 for candidate in raw["candidates"].values() for case in candidate["cases"])
