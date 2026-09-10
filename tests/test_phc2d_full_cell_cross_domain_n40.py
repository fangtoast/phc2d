"""Contract tests for the frozen N40 K20/50K cross-domain campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from phc2d.forward.full_cell_cross_domain_n40 import (
    DOMAINS,
    EXPECTED_CELLS,
    EXPECTED_CHECKPOINTS,
    MODELS,
    SHARDS,
    SHARD_OFFSETS,
    _metric_rows,
    build_evaluation_cells,
)


def _checkpoints() -> list[dict[str, object]]:
    rows = []
    for family, seeds in (("K20", (0,)), ("F50", (0, 1, 2))):
        for seed in seeds:
            for model in MODELS:
                for lane in ("B", "G"):
                    rows.append({
                        "source_family": family,
                        "model": model,
                        "source_lane": lane,
                        "seed": seed,
                        "checkpoint_path": f"runs/{family}/{model}_{lane}_{seed}.pt",
                        "checkpoint_sha256": f"{len(rows):064x}",
                    })
            rows.append({
                "source_family": family,
                "model": "dual_lane_band_operator_v1",
                "source_lane": "J",
                "seed": seed,
                "checkpoint_path": f"runs/{family}/joint_{seed}.pt",
                "checkpoint_sha256": f"{len(rows):064x}",
            })
    return rows


def test_exact_68_checkpoints_and_272_cells() -> None:
    checkpoints = _checkpoints()
    cells = build_evaluation_cells(checkpoints)
    assert len(checkpoints) == EXPECTED_CHECKPOINTS
    assert len(cells) == EXPECTED_CELLS
    assert len({row["evaluation_id"] for row in cells}) == EXPECTED_CELLS
    assert {row["target_domain"] for row in cells} == set(DOMAINS)
    assert all(sum(row["target_domain"] == domain for row in cells) == 68 for domain in DOMAINS)


def test_fixed_shards_are_3_3_1_1_and_cover_each_cell_once() -> None:
    cells = build_evaluation_cells(_checkpoints())
    for domain, count in SHARDS.items():
        selected = [row for row in cells if row["target_domain"] == domain]
        assert {int(row["shard_id"]) for row in selected} == set(range(SHARD_OFFSETS[domain], SHARD_OFFSETS[domain] + count))
        assert sum(int(row["shard_id"]) == SHARD_OFFSETS[domain] + shard for row in selected for shard in range(count)) == len(selected)
    assert all(sum(row["target_domain"] == domain and int(row["shard_id"]) == SHARD_OFFSETS[domain] + shard for row in cells) in ({68} if SHARDS[domain] == 1 else {22, 23}) for domain in DOMAINS for shard in range(SHARDS[domain]))
    assert {int(row["shard_id"]) for row in cells} == set(range(8))


def test_weighted_sharding_is_deterministic_and_splits_expensive_models() -> None:
    weights = {model: 1.0 for model in MODELS}
    weights.update({"fno2d_band_v2": 10.0, "tfno2d_band_v3": 20.0, "dual_lane_band_operator_v1": 30.0})
    first = build_evaluation_cells(_checkpoints(), weights)
    second = build_evaluation_cells(_checkpoints(), weights)
    assert [(row["evaluation_id"], row["shard_id"]) for row in first] == [(row["evaluation_id"], row["shard_id"]) for row in second]
    for domain in ("F50-B", "F50-G"):
        expensive = {int(row["shard_id"]) for row in first if row["target_domain"] == domain and row["source_model"] in {"fno2d_band_v2", "tfno2d_band_v3", "dual_lane_band_operator_v1"}}
        assert len(expensive) >= 2


def test_zero_variance_r2_is_undefined() -> None:
    target = np.ones((4, 2, 3), dtype=np.float32)
    target[:, 0, 1] = np.arange(4, dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0, 1] += 0.2
    metrics, _, per_coordinate = _metric_rows(prediction, target)
    assert metrics["undefined_r2_coordinates"] == 5
    assert any(row["r2_status"] == "undefined_zero_target_variance" for row in per_coordinate)
    assert np.isfinite(metrics["bands_2_10_r2_variance_weighted"])


def test_jobs_enforce_fixed_array_afterok_and_no_explicit_memory() -> None:
    root = Path(__file__).resolve().parents[1]
    submit = (root / "jobs/paracloud_submit_full_cell_cross_domain.sh").read_text(encoding="utf-8")
    worker = (root / "jobs/paracloud_full_cell_cross_domain.sh").read_text(encoding="utf-8")
    preflight = (root / "jobs/paracloud_full_cell_cross_domain_preflight.sh").read_text(encoding="utf-8")
    finalizer = (root / "jobs/paracloud_full_cell_cross_domain_finalize.sh").read_text(encoding="utf-8")
    assert "--array=0-7%8" in submit
    assert "--gres=gpu:1" in submit and "-c 6" in submit and "-t 04:00:00" in submit
    assert "--mem" not in submit
    assert '--dependency="afterok:$shard_job_id"' in submit
    assert "SLURM_ARRAY_TASK_ID" in worker
    assert "n40_profile.csv" in worker
    assert "preflight" in preflight and "SLURM_JOB_ID" not in preflight
    assert "full_cell_cross_domain_n40 finalize" in finalizer


def test_config_excludes_p4m_and_has_exact_domain_shard_contract() -> None:
    config = Path("configs/research/full_cell_cross_domain_n40_v1.yaml").read_text(encoding="utf-8")
    assert "include_p4m: false" in config
    assert "F50-B: 3" in config and "F50-G: 3" in config
    assert "K20-B: 1" in config and "K20-G: 1" in config
