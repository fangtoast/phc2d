"""Tests for guarded query data-scaling orchestration.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json

import numpy as np
import yaml

from phc2d.data import ForwardDatasetBundle, save_forward_dataset
from phc2d.query.runner import _query_per_band_rows
from phc2d.query.scaling import _child_config, build_scale_plan, run_query_data_scaling


def test_scale_plan_blocks_missing_exact_v2_sizes() -> None:
    raw = {"scaling": {"sizes": [5, 8], "require_exact_dataset_size": True}}
    audit = {"n_structures": 5}

    rows = build_scale_plan(raw, audit)

    assert rows[0]["status"] == "runnable"
    assert rows[1]["status"] == "pending_larger_v2_dataset"
    assert "only exact guarded V2 dataset N=5" in rows[1]["reason"]


def test_child_config_does_not_inherit_parent_run_dir(tmp_path) -> None:
    child = _child_config(
        {
            "experiment": {"experiment_id": "query_data_scaling_unit"},
            "dataset": {"path": "old.npz", "paths_by_scale": {"64": "new.npz"}},
            "output": {"runs_dir": "runs", "run_dir": "runs/20260619/parent_202606190000"},
        },
        scale=64,
        run_dir=tmp_path / "runs" / "parent",
        dataset_path="new.npz",
    )

    assert child["experiment"]["experiment_id"] == "query_data_scaling_unit_N64"
    assert child["dataset"]["path"] == "new.npz"
    assert "paths_by_scale" not in child["dataset"]
    assert "run_dir" not in child["output"]


def test_scale_plan_uses_scale_specific_dataset_paths() -> None:
    raw = {"scaling": {"sizes": [64, 256, 1000], "require_exact_dataset_size": True}}
    audit = {
        "available_by_scale": {
            "64": {"dataset_path": "data/query_dataset_v2_pilot_64_mpb.npz", "n_structures": 64},
            "256": {"dataset_path": "data/query_dataset_v2_pilot_256_mpb.npz", "n_structures": 256},
            "1000": {"dataset_path": "data/query_dataset_v2_pilot_1000_mpb.npz", "n_structures": 1000},
        }
    }

    rows = build_scale_plan(raw, audit)

    assert [row["status"] for row in rows] == ["runnable", "runnable", "runnable"]
    assert rows[1]["dataset_path"] == "data/query_dataset_v2_pilot_256_mpb.npz"
    assert rows[2]["dataset_path"] == "data/query_dataset_v2_pilot_1000_mpb.npz"


def test_scale_plan_can_select_only_n1000() -> None:
    raw = {
        "scaling": {
            "sizes": [64, 256, 1000],
            "only_sizes": [1000],
            "require_exact_dataset_size": True,
        }
    }
    audit = {
        "available_by_scale": {
            "64": {"dataset_path": "data/query_dataset_v2_pilot_64_mpb.npz", "n_structures": 64},
            "256": {"dataset_path": "data/query_dataset_v2_pilot_256_mpb.npz", "n_structures": 256},
            "1000": {"dataset_path": "data/query_dataset_v2_pilot_1000_mpb.npz", "n_structures": 1000},
        }
    }

    rows = build_scale_plan(raw, audit)

    assert [row["status"] for row in rows] == [
        "skipped_by_scale_filter",
        "skipped_by_scale_filter",
        "runnable",
    ]
    assert rows[2]["scale_n"] == 1000
    assert rows[2]["dataset_path"] == "data/query_dataset_v2_pilot_1000_mpb.npz"


def test_full_n1000_config_selects_only_n1000() -> None:
    raw = yaml.safe_load(
        open("configs/research/data_scaling_benchmark_v2_1000_full.yaml", encoding="utf-8")
    )
    rows = build_scale_plan(
        raw,
        {
            "available_by_scale": {
                "64": {"dataset_path": "data/query_dataset_v2_pilot_64_mpb.npz", "n_structures": 64},
                "256": {"dataset_path": "data/query_dataset_v2_pilot_256_mpb.npz", "n_structures": 256},
                "1000": {"dataset_path": "data/query_dataset_v2_pilot_1000_mpb.npz", "n_structures": 1000},
            }
        },
    )

    assert [row["status"] for row in rows] == [
        "skipped_by_scale_filter",
        "skipped_by_scale_filter",
        "runnable",
    ]
    assert raw.get("limits") is None
    assert raw["training"]["seeds"] == [0, 1, 2]
    assert raw["training"]["batch_size"] == 64


def test_full_n10000_config_selects_only_n10000_neural_boundary() -> None:
    raw = yaml.safe_load(
        open("configs/research/data_scaling_benchmark_v2_10000_full.yaml", encoding="utf-8")
    )
    rows = build_scale_plan(
        raw,
        {
            "available_by_scale": {
                "10000": {
                    "dataset_path": "data/query_dataset_v2_pilot_10000_mpb.npz",
                    "n_structures": 10000,
                },
            }
        },
    )

    assert [row["status"] for row in rows] == ["runnable"]
    assert rows[0]["scale_n"] == 10000
    assert raw["query"]["requires_physical_k_coordinates"] is False
    assert raw["query"]["boundary"] == "neural_surrogate_no_mpb_meep_generated_not_replayed"
    assert raw["training"]["seeds"] == [0, 1, 2]
    assert raw["training"]["batch_size"] == 64


def test_query_per_band_rows_are_non_empty() -> None:
    y_true = np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 3.0, 2.0], [1.0, 4.0, 10.0]], dtype=np.float32)

    rows = _query_per_band_rows(
        y_true,
        y_pred,
        case_id="unit_case",
        protocol="unseen_structure_all_k",
        model="mlp_query",
        seed=0,
    )

    assert len(rows) == 3
    assert rows[0]["band_one_based"] == 1
    assert rows[1]["query_mae"] == 0.5
    assert rows[2]["n_eval_queries"] == 2


def test_query_data_scaling_writes_blocked_artifacts_without_training(tmp_path) -> None:
    dataset_path = tmp_path / "query_v2_unit.npz"
    bundle = ForwardDatasetBundle(
        rho=np.zeros((5, 36), dtype=np.float32),
        Wh=np.ones((5, 4, 3), dtype=np.float32),
        idx_train=np.array([0, 1, 2], dtype=np.int64),
        idx_val=np.array([3], dtype=np.int64),
        idx_test=np.array([4], dtype=np.int64),
        metadata=[{"sample_id": f"unit_{idx:06d}", "split": "train" if idx < 3 else "test"} for idx in range(5)],
        config_yaml="k_path:\n  n_total: 4\n",
    )
    save_forward_dataset(dataset_path, bundle)
    config_path = tmp_path / "scaling.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "query_data_scaling_unit"},
                "dataset": {"path": str(dataset_path)},
                "scaling": {"sizes": [8], "require_exact_dataset_size": True},
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_query_data_scaling(config_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "blocked"
    assert metrics["completed_scales"] == []
    assert metrics["pending_scales"] == [8]
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "pending_scales.csv").is_file()


def test_query_data_scaling_reports_skipped_scales_separately(tmp_path) -> None:
    dataset_path = tmp_path / "query_v2_unit.npz"
    bundle = ForwardDatasetBundle(
        rho=np.zeros((5, 36), dtype=np.float32),
        Wh=np.ones((5, 4, 3), dtype=np.float32),
        idx_train=np.array([0, 1, 2], dtype=np.int64),
        idx_val=np.array([3], dtype=np.int64),
        idx_test=np.array([4], dtype=np.int64),
        metadata=[{"sample_id": f"unit_{idx:06d}", "split": "train" if idx < 3 else "test"} for idx in range(5)],
        config_yaml="k_path:\n  n_total: 4\n",
    )
    save_forward_dataset(dataset_path, bundle)
    config_path = tmp_path / "scaling.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "query_data_scaling_unit"},
                "dataset": {"path": str(dataset_path)},
                "scaling": {"sizes": [5, 8], "only_sizes": [8], "require_exact_dataset_size": True},
                "output": {"runs_dir": str(tmp_path / "runs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir = run_query_data_scaling(config_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "blocked"
    assert metrics["completed_scales"] == []
    assert metrics["skipped_scales"] == [5]
    assert metrics["pending_scales"] == [8]
