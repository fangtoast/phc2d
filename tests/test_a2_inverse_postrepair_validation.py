"""Tests for the frozen A2 post-repair validation lane.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from phc2d.inverse.benchmark import (
    POSTREPAIR_EVIDENCE_STATUS,
    SURROGATE_ABLATION_MODELS,
    _postrepair_gate_failures,
    _postrepair_variants,
    _require_checkpoint_allowed,
    _select_postrepair_checkpoint,
    run_a2_postrepair_pilot_replay,
    run_a2_postrepair_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "research" / "a2_inverse_surrogate_postrepair_validation_v2.yaml"


def _config() -> dict:
    return dict(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def test_postrepair_config_registers_seven_recipe_specific_checkpoint_grids() -> None:
    config = _config()
    variants = _postrepair_variants(config)

    assert tuple(item["id"] for item in variants) == SURROGATE_ABLATION_MODELS
    assert config["surrogate_matrix"]["reference_id"] == "fno2d_band_v2"
    assert config["protocol"]["forbid_test_access"] is True
    assert config["boundaries"]["test100_access_allowed"] is False
    assert config["boundaries"]["test625_access_allowed"] is False
    assert config["boundaries"]["pino_status"] == "blocked"
    assert POSTREPAIR_EVIDENCE_STATUS == "single_seed_20_target_pilot"
    for variant in variants:
        assert [int(item["epoch"]) for item in variant["generator_checkpoints"]] == [10, 30, 50]
        assert variant["training_config_sha256"]
        assert variant["primary_ranker"]["sha256"]


def test_withdrawn_checkpoint_is_rejected_by_hash_and_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "withdrawn.pt"
    checkpoint.write_bytes(b"withdrawn")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = {
        "withdrawn_checkpoint_denylist": {
            "sha256": [digest],
            "path_prefixes": [],
        }
    }
    with pytest.raises(ValueError, match="withdrawn and denied by SHA-256"):
        _require_checkpoint_allowed(
            config,
            checkpoint,
            digest,
            label="withdrawn test checkpoint",
        )

    config["withdrawn_checkpoint_denylist"] = {
        "sha256": [],
        "path_prefixes": [str(tmp_path)],
    }
    with pytest.raises(ValueError, match="withdrawn and denied by path"):
        _require_checkpoint_allowed(
            config,
            checkpoint,
            digest,
            label="withdrawn test checkpoint",
        )


def test_checkpoint_selection_uses_frozen_hierarchy_and_blocks_empty_health() -> None:
    rows = [
        {
            "epoch": 10,
            "gate_status": "pass",
            "selection_score_max_primary_common_mre": 0.03,
            "threshold_margin_fraction": 0.10,
            "sampling_plus_ranking_time_s": 3.0,
        },
        {
            "epoch": 30,
            "gate_status": "pass",
            "selection_score_max_primary_common_mre": 0.02,
            "threshold_margin_fraction": 0.12,
            "sampling_plus_ranking_time_s": 4.0,
        },
        {
            "epoch": 50,
            "gate_status": "pass",
            "selection_score_max_primary_common_mre": 0.02,
            "threshold_margin_fraction": 0.12,
            "sampling_plus_ranking_time_s": 4.0,
        },
    ]
    selected, status = _select_postrepair_checkpoint(rows)
    assert status == "selected"
    assert int(selected["epoch"]) == 30

    blocked_rows = [{**row, "gate_status": "blocked_candidate_health"} for row in rows]
    selected, status = _select_postrepair_checkpoint(blocked_rows)
    assert status == "blocked_candidate_health"
    assert int(selected["epoch"]) == 10


def test_candidate_health_rejects_cpu_fallback_nan_and_collapsed_pool() -> None:
    gates = _config()["checkpoint_selection"]["hard_gates"]
    row = {
        "validity_rate": 1.0,
        "effective_unique_k": 12.0,
        "effective_unique_k_p10": 4.0,
        "continuous_candidate_std": 0.01,
        "threshold_margin_fraction": 0.30,
        "latent_input_gradient_norm": float("nan"),
        "mode_coverage_fraction": 0.5,
        "training_metrics_finite": False,
        "resolved_device": "cpu",
        "device_fallback_reason": "cuda_oom",
        "validation_best_of_k_surrogate_mre": 0.02,
        "validation_primary_selected_common_fno_mre": 0.03,
        "candidate_sampling_time_s": 1.0,
        "primary_surrogate_ranking_time_s": 1.0,
        "common_fno_diagnostic_ranking_time_s": 1.0,
    }
    failures = _postrepair_gate_failures(row, gates=gates)
    assert any(item.startswith("effective_unique_k:") for item in failures)
    assert "latent_input_gradient_norm:non_finite" in failures
    assert "training_metrics:non_finite" in failures
    assert "resolved_device:cpu" in failures
    assert "device_fallback_reason:cuda_oom" in failures


def test_live_postrepair_preflight_verifies_physical_k_and_all_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    run_a2_postrepair_preflight(CONFIG, run_dir=tmp_path)
    manifest = json.loads((tmp_path / "postrepair_surrogate_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((tmp_path / "preflight_decision.json").read_text(encoding="utf-8"))

    assert manifest["target_count"] == 100
    assert manifest["k_query_mode"] == "physical_coordinates_only_no_index_fallback"
    assert manifest["repeated_gamma_endpoint_indices"] == [30]
    assert len(manifest["variants"]) == 7
    assert sum(len(item["generator_checkpoints"]) for item in manifest["variants"]) == 21
    assert decision["selection_authorized"] is True
    assert decision["real_mpb_smoke_authorized"] is False


def test_pilot_cannot_start_before_real_mpb_smoke_pass(tmp_path: Path) -> None:
    (tmp_path / "real_mpb_smoke_decision.json").write_text(
        json.dumps({"status": "blocked", "pilot_replay_authorized": False}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="blocked by the real-MPB smoke decision"):
        run_a2_postrepair_pilot_replay(CONFIG, run_dir=tmp_path)
