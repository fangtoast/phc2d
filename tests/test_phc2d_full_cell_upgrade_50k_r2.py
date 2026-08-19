"""Tests for the seed-0 Full-Wh upgrade 50k R2 supplement.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from phc2d.forward.full_cell_upgrade_50k_evaluation import (
    _evaluate_cell_r2,
    _sha256,
    _sha256_array,
    build_evaluation_cells,
    coordinate_r2_statistics,
    plan_r2_supplement,
    summarize_coordinate_r2,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sources() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_models = ("mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2")
    upgrade_models = (
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
    )
    for group, models in (("baseline40k_e200", baseline_models), ("upgrade50k", upgrade_models)):
        for model in models:
            for lane in ("B", "G"):
                for seed in (0, 1, 2):
                    rows.append(
                        {
                            "source_id": f"{group}_{model}_{lane}_s{seed}",
                            "source_group": group,
                            "model": model,
                            "head": "legacy" if group.startswith("baseline") else "dense",
                            "source_lane": lane,
                            "seed": seed,
                            "checkpoint_path": f"checkpoints/{group}_{model}_{lane}_s{seed}.pt",
                            "checkpoint_sha256": "a" * 64,
                            "source_run_dir": "runs/source",
                            "actual_stop_epoch": 200,
                            "best_validation_epoch": 100,
                            "train_time_s": 1.0,
                            "training_peak_gpu_memory_mib": 1.0,
                            "checkpoint_bytes": 10,
                            "git_commit": "abc1234",
                            "epoch_100_decision": "continue_to_max_200",
                        }
                    )
    for seed in (0, 1, 2):
        rows.append(
            {
                "source_id": f"upgrade50k_dual_lane_band_operator_v1_J_s{seed}",
                "source_group": "upgrade50k",
                "model": "dual_lane_band_operator_v1",
                "head": "dense",
                "source_lane": "J",
                "seed": seed,
                "checkpoint_path": f"checkpoints/dual_s{seed}.pt",
                "checkpoint_sha256": "b" * 64,
                "source_run_dir": "runs/source",
                "actual_stop_epoch": 200,
                "best_validation_epoch": 100,
                "train_time_s": 1.0,
                "training_peak_gpu_memory_mib": 1.0,
                "checkpoint_bytes": 10,
                "git_commit": "abc1234",
                "epoch_100_decision": "continue_to_max_200",
            }
        )
    return rows


def test_coordinate_r2_covers_perfect_mean_negative_and_constant_outputs() -> None:
    target = np.asarray([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
    perfect = coordinate_r2_statistics(target, target)
    assert perfect["r2"][0] == pytest.approx(1.0)
    assert bool(perfect["valid"][1]) is False
    mean_prediction = np.broadcast_to(np.mean(target, axis=0), target.shape)
    mean_stats = coordinate_r2_statistics(target, mean_prediction)
    assert mean_stats["r2"][0] == pytest.approx(0.0)
    worse = coordinate_r2_statistics(target, np.full_like(target, 10.0))
    assert worse["r2"][0] < 0.0


def test_variance_weighted_r2_centers_each_output_coordinate() -> None:
    target = np.asarray(
        [
            [[0.0, 100.0], [0.0, 200.0]],
            [[1.0, 101.0], [2.0, 202.0]],
            [[2.0, 102.0], [4.0, 204.0]],
        ]
    )
    prediction = target.copy()
    prediction[:, :, 1] += 1.0
    statistics_payload = coordinate_r2_statistics(target, prediction)
    full = summarize_coordinate_r2(statistics_payload, band_start_zero_based=0)
    second_band = summarize_coordinate_r2(statistics_payload, band_start_zero_based=1)
    assert full["variance_weighted_r2"] < 1.0
    assert second_band["defined_coordinate_count"] == 2
    assert second_band["undefined_coordinate_count"] == 0
    assert second_band["variance_weighted_r2"] == pytest.approx(0.4)


def test_r2_plan_selects_only_18_native_seed_zero_cells(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    sources = _sources()
    plan = build_evaluation_cells(sources, shard_count=1)
    _write_csv(parent / "evaluation_plan.csv", plan)
    _write_csv(parent / "checkpoint_inventory.csv", sources)
    _write_csv(
        parent / "evaluation_cells.csv",
        [{**row, "test_bands_2_10_mre_pct": 1.0, "test_full_matrix_mre_pct": 1.0} for row in plan],
    )
    (parent / "results.json").write_text(
        json.dumps({"status": "formal_upgrade_50k_evaluation_success"}), encoding="utf-8"
    )
    (parent / "config.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "plan_hash": _sha256(parent / "evaluation_plan.csv"),
                "datasets": {
                    "B": {"path": "binary.h5"},
                    "G": {"path": "graded.h5"},
                },
            }
        ),
        encoding="utf-8",
    )
    output = plan_r2_supplement(parent, output_dir=tmp_path / "r2", repository_root=tmp_path)
    selected = list(csv.DictReader((output / "evaluation_plan.csv").open(encoding="utf-8")))
    inventory = list(csv.DictReader((output / "checkpoint_inventory.csv").open(encoding="utf-8")))
    config = yaml.safe_load((output / "config.resolved.yaml").read_text(encoding="utf-8"))
    assert len(selected) == 18
    assert len(inventory) == 17
    assert {int(row["seed"]) for row in selected} == {0}
    assert all(row["contract"].startswith("native_") for row in selected)
    assert {int(row["evaluation_resolution"]) for row in selected} == {64}
    assert config["primary_metric"] == "bands_2_10_coordinate_centered_variance_weighted_r2"
    with pytest.raises(ValueError, match="seed=0"):
        plan_r2_supplement(parent, output_dir=tmp_path / "bad", repository_root=tmp_path, seed=1)


def test_r2_hdf5_evaluation_uses_denormalized_targets(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    torch = pytest.importorskip("torch")
    path = tmp_path / "tiny.h5"
    indices = np.arange(4, dtype=np.int64)
    k_points = np.zeros((2, 2), dtype=np.float32)
    values = np.arange(1, 5, dtype=np.float32)
    grids = np.broadcast_to(values[:, None, None], (4, 2, 2)).copy()
    targets = np.stack([values[:, None] * np.asarray([1.0, 2.0])] * 2, axis=1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("idx_test", data=indices)
        handle.create_dataset("k_points", data=k_points)
        handle.create_dataset("material_weight_grid_64", data=grids)
        handle.create_dataset("Wh", data=targets)

    class ExactModel(torch.nn.Module):
        def forward(self, xb: object) -> object:
            value = xb[:, 0]
            first = torch.stack((value, 2.0 * value), dim=-1)
            return torch.stack((first, first), dim=1)

    device_info = SimpleNamespace(
        torch_device=torch.device("cpu"),
        num_workers=0,
        pin_memory=False,
        is_cuda=False,
    )
    spec = SimpleNamespace(
        input_layout="flat",
        accepts_k_points=False,
        accepts_lane_id=False,
        name="exact_model",
    )
    metrics, coordinates, bands = _evaluate_cell_r2(
        model=ExactModel(),
        model_spec=spec,
        payload={
            "target_mean": torch.zeros((2, 2)),
            "target_std": torch.ones((2, 2)),
        },
        row={"evaluation_id": "tiny", "target_lane": "B", "source_lane": "B"},
        dataset_contract={
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "idx_test_sha256": _sha256_array(indices),
            "k_points_sha256": _sha256_array(k_points),
            "test_count": 4,
            "grid_fields": {64: "material_weight_grid_64"},
            "sampling": "continuous",
            "threshold": None,
        },
        root=tmp_path,
        device_info=device_info,
        torch=torch,
    )
    assert metrics["bands_2_10_variance_weighted_r2"] == pytest.approx(1.0)
    assert metrics["full_variance_weighted_r2"] == pytest.approx(1.0)
    assert metrics["test_bands_2_10_mre_pct"] == pytest.approx(0.0)
    assert len(coordinates) == 4
    assert len(bands) == 2
