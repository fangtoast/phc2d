"""Contract tests for the EXP-DAT-005 K=5 Xingyi dataset configurations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from phc2d.data.full_cell import build_full_bz_grid
from phc2d.data.full_cell_generation import _validate_config, load_full_cell_generation_config


ROOT = Path(__file__).resolve().parents[1]


def test_exp_dat_005_is_registered_with_a_reachable_protocol_and_configs() -> None:
    registry = yaml.safe_load((ROOT / "experiments/registry.yaml").read_text(encoding="utf-8"))
    entry = next(item for item in registry["experiments"] if item["id"] == "EXP-DAT-005")
    assert entry["status"] == "planned"
    assert entry["spec"] == ["experiments/research/full_cell_k5_dual_lane_xingyi.md"]
    assert entry["config"] == [
        "configs/research/full_cell_binary_fourier_k5_bz9_10000_xingyi_v1.yaml",
        "configs/research/full_cell_graded_fourier_k5_bz9_10000_xingyi_v1.yaml",
    ]


def test_exp_dat_005_configs_freeze_independent_k5_contract() -> None:
    paths = [
        ROOT / "configs/research/full_cell_binary_fourier_k5_bz9_10000_xingyi_v1.yaml",
        ROOT / "configs/research/full_cell_graded_fourier_k5_bz9_10000_xingyi_v1.yaml",
    ]
    configs = [load_full_cell_generation_config(path) for path in paths]
    assert all(_validate_config(config) == [] for config in configs)
    assert [config["experiment"]["experiment_id"] for config in configs] == [
        "full_cell_binary_fourier_k5_bz9_10000_xingyi_v1",
        "full_cell_graded_fourier_k5_bz9_10000_xingyi_v1",
    ]
    assert [config["dataset"]["seed"] for config in configs] == [2026082105, 2026082106]
    assert len({config["dataset"]["seed"] for config in configs}) == 2
    assert all("paired_coefficient_seed" not in config["generator"] for config in configs)
    for config in configs:
        assert config["dataset"]["protocol"] == "full_cell_fourier_v2"
        assert config["generator"]["fourier_cutoff"] == 5
        assert config["dataset"]["sample_count"] == 10_000
        assert config["dataset"]["split"] == {"train": 8000, "val": 1000, "test": 1000}
        assert config["dataset"]["raster_resolutions"] == [32, 64, 128, 256]
        assert config["solver"]["resolution"] == 64
        assert config["solver"]["num_bands"] == 10
        assert config["solver"]["band_guard_count"] == 4
        assert config["k_grid"]["n_side"] == 9
        assert len(build_full_bz_grid(config["k_grid"]["n_side"])[0]) == 81
        assert config["workers"]["workers"] == 64
        assert config["workers"]["min_workers"] == 1
        assert config["workers"]["max_workers"] == 64
        assert config["generator"]["plan_workers"] == 8

    assert configs[0]["dataset"]["lane"] == "binary_fourier"
    assert configs[1]["dataset"]["lane"] == "graded_fourier"
    assert configs[0]["dataset"]["output_path"] == (
        "data/full_cell_binary_fourier_k5_bz9_10000_xingyi_v1.h5"
    )
    assert configs[1]["dataset"]["output_path"] == (
        "data/full_cell_graded_fourier_k5_bz9_10000_xingyi_v1.h5"
    )
