"""Tests for the shared HPC physics campaign control plane.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from phc2d.cli import build_parser
from phc2d.hpc import (
    PhysicalCampaignManifest,
    PhysicalCase,
    PhysicalShardSpec,
    PlatformProfile,
    admitted_scaling_tiers,
    atomic_write_json,
    collect_campaign_status,
    get_runtime_adapter,
    list_adapter_specs,
    recommend_node_count,
    validate_shard_coverage,
)


def test_balanced_shards_cover_plan_exactly() -> None:
    specs = [
        PhysicalShardSpec.partition(
            case_count=50_003,
            shard_index=index,
            shard_count=4,
            workers=64,
            plan_sha256="a" * 64,
        )
        for index in range(4)
    ]
    validate_shard_coverage(specs, case_count=50_003)
    assert [(item.case_start, item.case_end) for item in specs] == [
        (0, 12_501),
        (12_501, 25_002),
        (25_002, 37_503),
        (37_503, 50_003),
    ]


def test_case_hash_is_stable_and_bad_coverage_fails() -> None:
    case = PhysicalCase("case-1", 0, {"rho": [1, 2]}, split="train", seed=7)
    assert case.to_mapping()["input_sha256"] == case.to_mapping()["input_sha256"]
    specs = [PhysicalShardSpec(0, 1, 1, 2, 1, "b" * 64)]
    with pytest.raises(ValueError, match="gap or overlap"):
        validate_shard_coverage(specs, case_count=2)


def test_resource_policy_selects_four_xingyi_nodes_for_full_cell_baseline() -> None:
    platform = PlatformProfile(
        platform_id="xingyi",
        scheduler="slurm",
        partition="xyfree",
        cores_per_node=64,
        memory_gb_per_node=128,
        max_nodes=8,
        max_array_tasks=16,
    )
    result = recommend_node_count(
        case_count=50_000,
        median_case_time_s=38.4,
        platform=platform,
    )
    assert result["selected_nodes"] == 4
    assert result["total_workers"] == 256
    assert result["target_met"] is True
    assert admitted_scaling_tiers({1: 1.66, 2: 3.0, 4: 5.2}) == (1, 2, 4)
    assert admitted_scaling_tiers({1: 1.66, 2: 3.0, 4: 4.0}) == (1, 2)


def test_all_physical_task_families_are_registered() -> None:
    adapters = {item.adapter_id: item for item in list_adapter_specs()}
    assert set(adapters) == {
        "field_eigen",
        "full_cell_band",
        "inverse_mpb_replay",
        "query_k_path",
        "resolution_convergence",
    }
    assert adapters["full_cell_band"].migration_status == "reference_adapter"
    assert get_runtime_adapter("full_cell_band").adapter_id == "full_cell_band"


def test_status_reads_one_progress_file_per_shard(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    atomic_write_json(
        run / "physical_plan_manifest.json",
        {
            "adapter_id": "full_cell_band",
            "case_count": 8,
            "plan_sha256": "c" * 64,
            "solver_fingerprint": "d" * 64,
            "output_schema": "test",
            "case_plan": "sample_plan.jsonl",
            "shard_count": 2,
        },
    )
    atomic_write_json(
        run / "physical_campaign_manifest.json",
        PhysicalCampaignManifest(
            campaign_id="unit",
            adapter_id="full_cell_band",
            state="plan_ready",
            plan_sha256="c" * 64,
            planned_cases=8,
            shard_count=2,
            platform_id="unit",
        ).to_mapping(),
    )
    for index in range(2):
        atomic_write_json(
            run / "shards" / f"shard_{index:03d}" / "progress.json",
            {
                "shard_index": index,
                "status": "completed",
                "completed": 4,
                "failed": 0,
                "recent_throughput_cases_per_s": 2.0,
                "updated_unix_s": 1.0,
            },
        )
    for index in range(1_000):
        (run / f"ignored_sample_{index}.tmp").write_text("x", encoding="utf-8")

    status = collect_campaign_status(run)

    assert status["completed"] == 8
    assert status["failed"] == 0
    assert status["state"] == "solver_shards_complete"
    assert status["recent_throughput_cases_per_s"] == 4.0


def test_hpc_cli_contract_is_lazy() -> None:
    parser = build_parser()
    plan = parser.parse_args(
        [
            "hpc",
            "plan",
            "--adapter",
            "full_cell_band",
            "--config",
            "config.yaml",
            "--run-dir",
            "run",
            "--workers",
            "64",
            "--shards",
            "4",
            "--platform",
            "xingyi",
        ]
    )
    assert plan.command == "hpc plan"
    assert plan.shards == 4
    benchmark = parser.parse_args(
        [
            "hpc",
            "benchmark",
            "--platform-profile",
            "platform.yaml",
            "--evidence",
            "evidence.json",
            "--case-count",
            "50000",
        ]
    )
    assert benchmark.command == "hpc benchmark"


def test_platform_profile_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "platform.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "platform_id": "unit",
                "scheduler": "slurm",
                "partition": "cpu",
                "cores_per_node": 8,
                "memory_gb_per_node": 16,
                "max_nodes": 4,
                "max_array_tasks": 8,
                "allowed_node_tiers": [1, 2, 4],
            }
        ),
        encoding="utf-8",
    )
    from phc2d.hpc import load_platform_profile

    assert load_platform_profile(path).allowed_node_tiers == (1, 2, 4)
    assert json.loads(json.dumps(load_platform_profile(path).to_mapping()))["cores_per_node"] == 8


def test_repository_platform_profiles_and_slurm_pipeline_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    from phc2d.hpc import load_platform_profile

    xingyi = load_platform_profile(root / "configs/research/hpc_platform_xingyi_v1.yaml")
    m9 = load_platform_profile(root / "configs/research/hpc_platform_paracloud_m9_v1.yaml")
    assert (xingyi.partition, xingyi.cores_per_node) == ("xyfree", 64)
    assert (m9.partition, m9.cores_per_node) == ("amd_m9_768", 256)
    stage = (root / "jobs/hpc_physics_stage.sh").read_text(encoding="utf-8")
    submit = (root / "jobs/hpc_submit_physics_campaign.sh").read_text(encoding="utf-8")
    assert "SLURM_ARRAY_TASK_ID" in stage
    assert "PHC2D_AUTHORIZE_FORMAL=YES" in stage
    assert "--dependency=afterok:" in submit
    assert "--array=" in submit
    assert "PHC2D_SUBMIT_MODE:-dry-run" in submit
