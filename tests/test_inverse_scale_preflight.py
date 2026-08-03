"""Contract tests for non-executing inverse and data-scale preflight artifacts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
INVERSE_PREFLIGHT = ROOT / "configs/research/inverse_paired_100_preflight_v1.yaml"
SCALE_PREFLIGHT = ROOT / "configs/research/scale_100k_governance_preflight_v1.yaml"
COST_TEMPLATE = ROOT / "configs/research/templates/scale_100_sample_cost_record_v1.csv"
TARGET_MANIFEST = ROOT / "configs/research/target_manifests/inverse_paired_test100_v1.json"


def test_inverse_preflight_freezes_fair_budget_and_stays_nonexecuting() -> None:
    preflight = yaml.safe_load(INVERSE_PREFLIGHT.read_text(encoding="utf-8"))
    manifest = json.loads(TARGET_MANIFEST.read_text(encoding="utf-8"))

    assert preflight["protocol"]["execution"] == "protocol_only_no_training_sampling_or_replay"
    assert preflight["protocol"]["status"] == "blocked_pending_b4_dense_k_and_separate_inverse_authorization"
    assert preflight["dataset"]["expected_sha256"] == manifest["dataset_sha256"]
    assert preflight["dataset"]["expected_target_count"] == len(manifest["target_indices"]) == 100
    assert preflight["comparison"]["seeds"] == [0, 1, 2]
    assert preflight["comparison"]["raw_candidates_per_target"] == 128
    assert preflight["comparison"]["replay_top_m_per_target"] == 5
    assert preflight["comparison"]["mpb_solves_per_method_per_seed"] == 500
    assert preflight["dataset"]["headline_band_range_one_based"] == [2, 10]
    assert "auto_or_implicit_checkpoint_selection" in preflight["lineage"]["reject_if"]
    assert preflight["boundaries"]["inverse_status"] == "blocked"


def test_scale_preflight_freezes_governance_without_generation() -> None:
    preflight = yaml.safe_load(SCALE_PREFLIGHT.read_text(encoding="utf-8"))

    assert preflight["governance"]["execution"] == "protocol_only_no_dataset_generation"
    assert preflight["governance"]["scope"] == "band_only_data_governance_not_full_field_generation"
    assert set(preflight["ood_strata"]["pools"]) == {
        "reference_id",
        "continuous",
        "fragmented",
        "boundary_hard",
    }
    assert preflight["deduplication"]["near_duplicate_metric"] == (
        "normalized_hamming_distance_over_geometry_dof_count"
    )
    assert preflight["deduplication"]["near_duplicate_if"] == "distance_less_than_0.01"
    assert "structure_sha256" in preflight["metadata_schema"]["required_fields"]
    assert "solver_provenance_sha256" in preflight["metadata_schema"]["required_fields"]
    assert preflight["cost_calibration"]["sample_count"] == 100
    assert preflight["launch_decisions"]["default_before_evidence"] == "defer"
    assert "dataset_generation_from_this_preflight_config" in preflight["launch_decisions"]["prohibit"]


def test_cost_record_template_carries_solver_and_lineage_fields() -> None:
    with COST_TEMPLATE.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    assert "wall_time_s" in header
    assert "solver_time_s" in header
    assert "failure_reason" in header
    assert "k_path_sha256" in header
    assert "structure_sha256" in header
    assert "solver_provenance_sha256" in header
