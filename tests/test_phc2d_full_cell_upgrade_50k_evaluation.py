"""Tests for frozen Full-Wh upgrade 50k post-training evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from phc2d.forward.full_cell_upgrade_50k_evaluation import (
    DIRECT_BASELINES,
    build_evaluation_cells,
    finalize_evaluation_campaign,
    finalize_threshold_diagnostic,
    plan_threshold_diagnostic,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sources() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in DIRECT_BASELINES:
        lanes = ("J",) if model == "dual_lane_band_operator_v1" else ("B", "G")
        for lane in lanes:
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "source_id": f"up_{model}_{lane}_{seed}",
                        "source_group": "upgrade50k",
                        "model": model,
                        "head": "dense",
                        "source_lane": lane,
                        "seed": seed,
                        "checkpoint_path": f"upgrade/{model}/{lane}/{seed}/checkpoint.pt",
                        "checkpoint_sha256": "a" * 64,
                        "source_run_dir": f"upgrade/{model}/{lane}/{seed}",
                        "actual_stop_epoch": 100,
                        "best_validation_epoch": 90,
                        "train_time_s": 1.0,
                        "training_peak_gpu_memory_mib": 1000.0,
                        "checkpoint_bytes": 100,
                        "git_commit": "commit",
                        "epoch_100_decision": "stop_at_100",
                    }
                )
    for model in set(DIRECT_BASELINES.values()):
        for lane in ("B", "G"):
            for seed in (0, 1, 2):
                rows.append(
                    {
                        "source_id": f"bl_{model}_{lane}_{seed}",
                        "source_group": "baseline40k_e200",
                        "model": model,
                        "head": "legacy",
                        "source_lane": lane,
                        "seed": seed,
                        "checkpoint_path": f"baseline/{model}/{lane}/{seed}/model.pt",
                        "checkpoint_sha256": "b" * 64,
                        "source_run_dir": f"baseline/{model}/{lane}/{seed}",
                        "actual_stop_epoch": 200,
                        "best_validation_epoch": "",
                        "train_time_s": 1.0,
                        "training_peak_gpu_memory_mib": 1000.0,
                        "checkpoint_bytes": 100,
                        "git_commit": "historical",
                        "epoch_100_decision": "historical_e200_budget",
                    }
                )
    return rows


def test_evaluation_matrix_freezes_native_cross_resolution_and_na_cells() -> None:
    cells = build_evaluation_cells(_sources(), shard_count=8)
    planned = [row for row in cells if row["planned_status"] == "planned"]
    not_applicable = [row for row in cells if row["planned_status"] != "planned"]
    assert len(planned) == 126
    assert len(not_applicable) == 18
    assert len({row["evaluation_id"] for row in cells}) == len(cells)
    assert {int(row["shard_id"]) for row in planned} == set(range(8))
    assert all(
        row["model"] != "res_mlp_gated_v2" or int(row["evaluation_resolution"]) == 64
        for row in planned
    )


def _value(row: dict[str, object]) -> float:
    model = str(row["model"])
    lane = str(row["target_lane"])
    if row["source_group"] == "baseline40k_e200":
        return 1.0
    if row["contract"].startswith("native"):
        if model == "mlp_mixer_band_v1":
            return 0.85 if lane == "B" else 1.02
        if model == "tfno2d_band_v3":
            return 0.90
        if model == "dual_lane_band_operator_v1":
            return 0.91
        return 1.01
    return 1.2


def _materialize_single_shard(campaign: Path, cells: list[dict[str, object]]) -> None:
    shard = campaign / "shards" / "shard_00"
    shard.mkdir(parents=True)
    completed = []
    band_rows = []
    ids = []
    indices = []
    samples = []
    for row in cells:
        value = _value(row)
        completed.append(
            {
                **row,
                "test_bands_2_10_mre_pct": value,
                "test_full_matrix_mre_pct": value,
                "test_mae": value,
                "test_rmse": value,
                "test_ordering_violation_rate": 0.0,
                "n_test": 4,
                "eval_time_s": 0.1,
                "inference_time_per_sample_ms": 25.0,
                "throughput_samples_per_s": 40.0,
                "status": "completed",
                "fine_tuned": False,
                "parameter_count": 3_000_000 if row["model"] == "dual_lane_band_operator_v1" else 2_500_000,
            }
        )
        band_rows.extend(
            {"evaluation_id": row["evaluation_id"], "band_one_based": band, "mre_pct": value}
            for band in range(1, 11)
        )
        ids.append(row["evaluation_id"])
        indices.append(np.arange(4, dtype=np.int64))
        samples.append(np.full(4, value, dtype=np.float64))
    _write_csv(shard / "evaluation_cells.csv", completed)
    _write_csv(shard / "history.csv", completed)
    _write_csv(shard / "per_band_metrics.csv", band_rows)
    np.savez_compressed(
        shard / "sample_metrics.npz",
        evaluation_ids=np.asarray(ids),
        dataset_indices=np.stack(indices),
        sample_mre_pct=np.stack(samples),
    )
    (shard / "config.resolved.yaml").write_text("shard_id: 0\n", encoding="utf-8")
    (shard / "run.log").write_text("complete\n", encoding="utf-8")
    (shard / "run.meta.json").write_text("{}\n", encoding="utf-8")
    (shard / "metrics.json").write_text(
        json.dumps({"peak_gpu_memory_mb": 900.0, "wall_time_total_s": 2.0, "eval_time_total_s": 1.0}),
        encoding="utf-8",
    )
    (shard / "results.json").write_text(
        json.dumps(
            {
                "status": "formal_upgrade_50k_evaluation_shard_success",
                "plan_hash": "test-plan",
                "completed_cells": len(cells),
            }
        ),
        encoding="utf-8",
    )
    (shard / "n40_profile.csv").write_text("timestamp,gpu_uuid\nnow,gpu0\n", encoding="utf-8")
    (shard / "n40_profile.json").write_text(
        json.dumps({"status": "profile_collected", "memory_used_mib_max": 1000.0}), encoding="utf-8"
    )


def test_finalizer_applies_promotion_and_evidence_boundaries(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    all_cells = build_evaluation_cells(_sources(), shard_count=1)
    planned = [row for row in all_cells if row["planned_status"] == "planned"]
    _write_csv(campaign / "evaluation_plan.csv", all_cells)
    (campaign / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "shard_count": 1,
                "bootstrap_replicates": 100,
                "plan_hash": "test-plan",
                "datasets": {"G": {"path": "graded.h5"}},
            }
        ),
        encoding="utf-8",
    )
    _write_csv(campaign / "checkpoint_inventory.csv", _sources())
    _materialize_single_shard(campaign, planned)
    finalize_evaluation_campaign(campaign, repository_root=tmp_path)
    results = json.loads((campaign / "results.json").read_text(encoding="utf-8"))
    decisions = list(csv.DictReader((campaign / "promotion_decisions.csv").open(encoding="utf-8")))
    statuses = {row["model"]: row["status"] for row in decisions}
    assert statuses["mlp_mixer_band_v1"] == "promoted_over_direct_baseline"
    assert results["formal_completed"] is True
    assert results["cross_lane_evidence_role"] == "domain_shift_diagnostic"
    assert results["resolution_evidence_role"].startswith("zero_shot_input_resolution")
    assert len(list(csv.DictReader((campaign / "native_test_summary.csv").open(encoding="utf-8")))) == 18
    assert len(list(csv.DictReader((campaign / "paired_bootstrap_comparisons.csv").open(encoding="utf-8")))) == 20


def _materialize_threshold_shards(threshold: Path) -> None:
    rows = list(csv.DictReader((threshold / "threshold_plan.csv").open(encoding="utf-8")))
    for shard_id in range(3):
        selected = [row for row in rows if int(row["shard_id"]) == shard_id]
        shard = threshold / "shards" / f"shard_{shard_id:02d}"
        shard.mkdir(parents=True)
        completed = []
        for row in selected:
            value = 0.8 + float(row["threshold"])
            completed.append(
                {
                    **row,
                    "test_bands_2_10_mre_pct": value,
                    "test_full_matrix_mre_pct": value,
                    "test_mae": value,
                    "test_rmse": value,
                    "test_ordering_violation_rate": 0.0,
                    "n_test": 4,
                    "eval_time_s": 0.1,
                    "inference_time_per_sample_ms": 25.0,
                    "throughput_samples_per_s": 40.0,
                    "status": "completed",
                    "fine_tuned": False,
                    "parameter_count": 2_500_000,
                }
            )
        _write_csv(shard / "evaluation_cells.csv", completed)
        _write_csv(shard / "history.csv", completed)
        _write_csv(
            shard / "per_band_metrics.csv",
            [{"evaluation_id": row["evaluation_id"], "band_one_based": 1, "mre_pct": 1.0} for row in selected],
        )
        np.savez_compressed(
            shard / "sample_metrics.npz",
            evaluation_ids=np.asarray([row["evaluation_id"] for row in selected]),
            dataset_indices=np.stack([np.arange(4) for _ in selected]),
            sample_mre_pct=np.stack([np.ones(4) for _ in selected]),
        )
        (shard / "run.log").write_text("complete\n", encoding="utf-8")
        (shard / "run.meta.json").write_text("{}\n", encoding="utf-8")
        (shard / "config.resolved.yaml").write_text(f"shard_id: {shard_id}\n", encoding="utf-8")
        (shard / "metrics.json").write_text(
            json.dumps({"peak_gpu_memory_mb": 800.0, "wall_time_total_s": 1.0}), encoding="utf-8"
        )
        config = yaml.safe_load((threshold / "config.resolved.yaml").read_text(encoding="utf-8"))
        (shard / "results.json").write_text(
            json.dumps(
                {
                    "status": "formal_threshold_diagnostic_shard_success",
                    "plan_hash": config["plan_hash"],
                }
            ),
            encoding="utf-8",
        )
        (shard / "n40_profile.csv").write_text("timestamp,gpu_uuid\nnow,gpu0\n", encoding="utf-8")
        (shard / "n40_profile.json").write_text(
            json.dumps({"status": "profile_collected", "memory_used_mib_max": 800.0}), encoding="utf-8"
        )


def test_threshold_diagnostic_runs_only_frozen_graded_winner(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    all_cells = build_evaluation_cells(_sources(), shard_count=1)
    _write_csv(campaign / "evaluation_plan.csv", all_cells)
    (campaign / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "shard_count": 1,
                "bootstrap_replicates": 100,
                "plan_hash": "test-plan",
                "datasets": {"G": {"path": "graded.h5"}},
            }
        ),
        encoding="utf-8",
    )
    _write_csv(campaign / "checkpoint_inventory.csv", _sources())
    _materialize_single_shard(campaign, [row for row in all_cells if row["planned_status"] == "planned"])
    finalize_evaluation_campaign(campaign, repository_root=tmp_path)
    threshold = plan_threshold_diagnostic(campaign, repository_root=tmp_path)
    plan = list(csv.DictReader((threshold / "threshold_plan.csv").open(encoding="utf-8")))
    assert len(plan) == 15
    assert {row["model"] for row in plan} == {"tfno2d_band_v3"}
    assert {float(row["threshold"]) for row in plan} == {0.3, 0.4, 0.5, 0.6, 0.7}
    _materialize_threshold_shards(threshold)
    finalize_threshold_diagnostic(threshold, repository_root=tmp_path)
    result = json.loads((threshold / "results.json").read_text(encoding="utf-8"))
    assert result["status"] == "formal_threshold_representation_proxy_success"
    assert result["evidence_role"] == "representation_proxy_not_physical_equivalence"
    report = (campaign / "report.md").read_text(encoding="utf-8")
    assert "Native held-out test 主结果" in report
    assert "逐 Band 结果" in report
    assert "Cross-lane 诊断" in report
    assert "32/128 零样本输入分辨率诊断" in report
    assert "Threshold Proxy" in report
