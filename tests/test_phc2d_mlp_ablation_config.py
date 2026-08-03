"""Contract tests for the A1.1 plain-MLP residual-connection ablation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from phc2d.forward import load_forward_run_config


def test_a1_forward_mlp_ablation_is_an_independent_single_model_protocol() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "research"
        / "a1_forward_mlp_ablation_v1.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = load_forward_run_config(config_path)

    assert raw["experiment"]["experiment_id"] == "a1_forward_mlp_ablation_v1"
    assert raw["fair_selection"]["enabled"] is True
    assert raw["fair_selection"]["expected_dataset_sha256"] == (
        "991cb3810e881938755495ad0bccfa5b091194c2b74451842a9a50ddce974ad4"
    )
    assert raw["fair_selection"]["expected_split_sizes"] == {
        "train": 6875,
        "val": 2500,
        "test": 625,
    }
    assert raw["benchmark"] == [
        {
            "input_repr": "rho_wedge",
            "output_target": "Wh",
            "model": "mlp",
            "model_params": {
                "hidden_dim": 256,
                "depth": 6,
                "dropout": 0.05,
                "activation": "gelu",
            },
        }
    ]
    assert config.epochs == 200
    assert config.batch_size == 64
    assert config.seeds == (0, 1, 2)
    assert config.learning_rates == (0.001, 0.0003, 0.0001)
    assert config.runtime.device == "cuda:0"
    assert config.runtime.require_accelerator is True
    assert config.runtime.precision == "fp32"
    assert config.runtime.amp_enabled is False
    assert config.runtime.tf32_enabled is False
