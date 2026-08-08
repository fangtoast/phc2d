"""Contract tests for the frozen 50k full-cell post-training evaluation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from phc2d.forward.full_cell_50k_evaluation import (
    EXPECTED_EXECUTABLE_CELLS,
    EXPECTED_NOT_APPLICABLE_CELLS,
    EXPECTED_TOTAL_CELLS,
    build_evaluation_cells,
)
from phc2d.forward.full_cell_wave import MODELS, SEEDS


def _source(
    group: str,
    representation: str,
    model: str,
    seed: int,
    *,
    epoch: int = 100,
    resolution: int = 64,
) -> dict[str, object]:
    identity = f"{group}-{representation}-{model}-{seed}-{epoch}-{resolution}"
    return {
        "source_id": identity,
        "source_group": group,
        "source_dataset_size": 40000 if group == "current40k" else 10000,
        "source_representation": representation,
        "model": model,
        "seed": seed,
        "epoch_budget": epoch,
        "training_resolution": resolution,
        "checkpoint_role": "best_validation",
        "checkpoint_path": f"runs/{identity}/model.pt",
        "checkpoint_sha256": identity.ljust(64, "0")[:64],
        "run_dir": f"runs/{identity}",
        "selection_split": "validation",
    }


def _all_sources() -> list[dict[str, object]]:
    sources = [
        _source("legacy10k_main", representation, model, seed)
        for representation in ("binary", "continuous", "t03", "t04", "t05", "t06", "t07")
        for model in MODELS
        for seed in SEEDS
    ]
    sources.extend(
        _source("legacy10k_ablation", representation, "fno2d_band_v2", seed, resolution=resolution)
        for representation in ("binary", "continuous", "t06")
        for resolution in (32, 128)
        for seed in SEEDS
    )
    sources.extend(
        _source("current40k", representation, model, seed, epoch=epoch)
        for representation in ("binary", "continuous")
        for epoch in (20, 100, 200)
        for model in MODELS
        for seed in SEEDS
    )
    return sources


def test_evaluation_plan_has_frozen_counts_and_no_duplicate_ids() -> None:
    rows = build_evaluation_cells(_all_sources(), shard_count=8)
    assert len(rows) == EXPECTED_TOTAL_CELLS
    assert len({row["evaluation_id"] for row in rows}) == EXPECTED_TOTAL_CELLS
    assert sum(row["planned_status"] == "planned" for row in rows) == EXPECTED_EXECUTABLE_CELLS
    assert sum(row["planned_status"] != "planned" for row in rows) == EXPECTED_NOT_APPLICABLE_CELLS
    assert {int(row["shard_id"]) for row in rows if row["planned_status"] == "planned"} == set(range(8))


def test_50k_final_fit_is_excluded_and_resolution_shape_failures_are_explicit() -> None:
    rows = build_evaluation_cells(_all_sources(), shard_count=8)
    assert all(row["source_group"] != "current50k_final_fit" for row in rows)
    not_applicable = [row for row in rows if row["planned_status"] != "planned"]
    assert {row["model"] for row in not_applicable} == {"mlp", "res_mlp_wide"}
    assert {int(row["evaluation_resolution"]) for row in not_applicable} == {32, 128}
    assert all(row["contracts"] == "zero_shot_input_resolution_transfer" for row in not_applicable)


def test_source_bundles_are_not_split_across_gpu_shards() -> None:
    rows = build_evaluation_cells(_all_sources(), shard_count=8)
    by_source: dict[str, set[int]] = {}
    for row in rows:
        if row["planned_status"] != "planned":
            continue
        by_source.setdefault(str(row["source_id"]), set()).add(int(row["shard_id"]))
    assert by_source
    assert all(len(shards) == 1 for shards in by_source.values())
