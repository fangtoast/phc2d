"""Validate the frozen A0/B0 research contracts without requiring local datasets or runs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PROTOCOL = ROOT / "configs/research/baseline_protocol_v1.yaml"
FIELD_CONTRACT = ROOT / "configs/research/field_eigen_te_contract.yaml"
TARGET_MANIFEST = ROOT / "configs/research/target_manifests/inverse_paired_test100_v1.json"
PHASE_DOCUMENT = ROOT / "docs/planning/0718-0801/09_phase1_a0_b0_protocol.md"


def test_baseline_protocol_freezes_shared_inverse_budget_and_manifest() -> None:
    protocol = yaml.safe_load(BASELINE_PROTOCOL.read_text(encoding="utf-8"))
    manifest = json.loads(TARGET_MANIFEST.read_text(encoding="utf-8"))

    assert protocol["protocol"]["status"] == "frozen_for_a1_a2"
    assert protocol["protocol"]["execution"] == "protocol_only_no_training_or_replay"
    assert protocol["dataset"]["sha256"] == manifest["dataset_sha256"]
    assert protocol["dataset"]["split_sizes"]["test"] == manifest["source_target_count"]
    assert protocol["inverse"]["target_manifest_path"] == "configs/research/target_manifests/inverse_paired_test100_v1.json"
    assert protocol["inverse"]["target_count"] == len(manifest["target_indices"]) == 100
    assert len(set(manifest["target_indices"])) == 100
    assert manifest["strata_counts"] == {
        "low_surrogate_mre": 25,
        "median_surrogate_mre": 25,
        "high_surrogate_mre": 25,
        "fixed_random": 25,
    }
    assert protocol["inverse"]["n_raw_candidates_per_target"] == 128
    assert protocol["inverse"]["replay_top_m_per_target"] == 5
    assert protocol["inverse"]["mpb_solves_per_method_per_seed"] == 500
    assert protocol["dataset"]["metric_domain"]["headline_band_range_one_based"] == [2, 10]


def test_field_eigen_contract_records_passed_traceability_and_b3_scope() -> None:
    contract = yaml.safe_load(FIELD_CONTRACT.read_text(encoding="utf-8"))
    fields = set(contract["record_schema"]["required_fields"])

    assert contract["contract"]["status"] == "b3_supervised_and_adapted_pino_gates_passed"
    assert contract["contract"]["version"] == 6
    assert contract["contract"]["execution"] == "b3_9216_operator_data_plus_supervised_v2_plus_adapted_pino_three_seed"
    assert contract["task"]["task_type"] == "bloch_eigenfield"
    assert contract["task"]["polarization"] == "TE"
    assert contract["task"]["band_selection"]["range_one_based"] == [2, 10]
    assert {"source_split", "eigenfrequency", "efield_complex", "hfield_complex", "bloch_phase", "dx", "dy", "field_normalization", "solver_provenance"} <= fields
    assert contract["record_schema"]["tensor_shape_policy"] == "derive_from_serialized_field_schema"
    assert contract["physics"]["boundary_condition"] == "bloch_periodic"
    assert contract["b1_handoff"]["initial_real_sample_count"] == 64
    assert {"pino_training", "maxwellnet_training", "field_inverse_integration"} <= set(contract["b1_handoff"]["prohibited_before_gate"])
    assert contract["b3_handoff"]["expected_record_count"] == 9216
    assert contract["b3_handoff"]["allowed_training_regimes"] == ["data_only_v2", "pino_adapted"]
    assert contract["b3_handoff"]["data_only_v2_gate"]["status"] == "passed"
    assert contract["b3_handoff"]["pino_adapted_gate"]["status"] == "pino_field_adapted_pilot_pass"
    assert contract["b3_handoff"]["pino_adapted_gate"]["selected_lambda_pde"] == 0.01
    assert contract["b3_handoff"]["pino_adapted_gate"]["residual_median_reduction_pct"] >= 20.0


def test_phase_document_links_both_frozen_contracts_and_preserves_claim_boundary() -> None:
    text = PHASE_DOCUMENT.read_text(encoding="utf-8")

    assert "baseline_protocol_v1.yaml" in text
    assert "field_eigen_te_contract.yaml" in text
    assert "不是跨方法或三 seed 的公平 superiority 结论" in text
    assert "PINO/MaxwellNet" in text
    assert "MPB replay" in text
