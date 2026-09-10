"""Tests for EXP-FWD-014 paired screen summarization.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

from phc2d.forward.gap_aware_loss import BASE_LOSS_NAME, GAP_AWARE_LOSS_NAME
from phc2d.forward.gap_screen import summarize_gap_screen


def _write_run(path: Path, loss_name: str, *, scale: float, joint: float) -> None:
    path.mkdir(parents=True)
    lane_metrics = {
        "band_accuracy": joint,
        "upper_edge_k_accuracy": joint,
        "joint_band_k_accuracy": joint,
        "selected_frequency_mre_pct": 2.0 * scale,
        "gap_width_mae": 0.02 * scale,
        "gap_center_mae": 0.01 * scale,
    }
    metrics = {
        "binary_validation_bands_2_10_mre_pct": 1.0,
        "graded_validation_bands_2_10_mre_pct": 1.0,
        "gap_validation": {"binary": lane_metrics, "graded": lane_metrics},
    }
    meta = {
        "dataset_sha256": {"binary": "a", "graded": "b"},
        "idx_train_hash": {"binary": "c", "graded": "d"},
        "idx_val_hash": {"binary": "e", "graded": "f"},
        "model": "dual_lane_band_operator_v1",
        "seed": 0,
    }
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (path / "results.json").write_text(
        json.dumps({"test_executed": False, "loss_contract": {"name": loss_name}}),
        encoding="utf-8",
    )
    (path / "run.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_gap_screen_passes_only_when_both_lanes_clear_the_registered_gate(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(baseline, BASE_LOSS_NAME, scale=1.0, joint=0.50)
    _write_run(candidate, GAP_AWARE_LOSS_NAME, scale=0.90, joint=0.53)
    out = summarize_gap_screen(baseline, candidate, tmp_path / "comparison")
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert results["status"] == "gap_screen_pass"
    assert results["test_executed"] is False
    assert metrics["lane_comparison"]["binary"]["passed"] is True
