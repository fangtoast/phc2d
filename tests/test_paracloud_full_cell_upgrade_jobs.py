"""Contract tests for upgraded Full-Wh N40 wave configs and scripts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "jobs" / "paracloud_submit_full_cell_upgrade_wave.sh"


def test_upgrade_10k_configs_freeze_models_data_and_balanced_lane_batch() -> None:
    binary = yaml.safe_load(
        (ROOT / "configs/research/full_cell_forward_upgrade_binary_10k_v1.yaml").read_text(encoding="utf-8")
    )
    graded = yaml.safe_load(
        (ROOT / "configs/research/full_cell_forward_upgrade_graded_10k_v1.yaml").read_text(encoding="utf-8")
    )
    dual = yaml.safe_load(
        (ROOT / "configs/research/full_cell_forward_upgrade_dual_lane_10k_v1.yaml").read_text(encoding="utf-8")
    )
    expected_models = {
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
    }
    assert set(binary["models"]) == expected_models
    assert set(graded["models"]) == expected_models
    assert binary["training"]["micro_batch_size"] == 64
    assert graded["training"]["micro_batch_size"] == 64
    assert dual["training"]["lane_batch_size"] == {"binary": 32, "graded": 32}
    assert dual["model"]["name"] == "dual_lane_band_operator_v1"
    assert dual["datasets"]["binary"]["expected_dataset_sha256"] == binary["dataset"]["expected_dataset_sha256"]
    assert dual["datasets"]["graded"]["expected_dataset_sha256"] == graded["dataset"]["expected_dataset_sha256"]


def test_submitter_freezes_unique_admission_and_10k_wave_tasks() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    expected = {
        "A0": 8,
        "A1": 8,
        "A2": 2,
        "10K-W1": 8,
        "10K-W2": 8,
        "10K-W3": 8,
        "10K-W4": 8,
        "10K-W5": 8,
        "10K-W6": 8,
        "10K-W7": 6,
    }
    all_formal: list[str] = []
    all_admission: list[str] = []
    for wave, count in expected.items():
        match = re.search(rf"^  {re.escape(wave)}\) specs=\(([^)]*)\)", text, flags=re.MULTILINE)
        assert match is not None
        tasks = match.group(1).split()
        assert len(tasks) == count
        assert len(set(tasks)) == count
        (all_admission if wave.startswith("A") else all_formal).extend(tasks)
    assert len(all_admission) == 18
    assert len(set(all_admission)) == 18
    assert len(all_formal) == 54
    assert len(set(all_formal)) == 54


def test_submitter_is_dry_run_by_default_and_requires_explicit_authorization() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    assert 'mode="${1:-}"' in text
    assert 'PHC2D_AUTHORIZE_UPGRADE_CAMPAIGN:-' in text
    assert 'PHC2D_EXPECTED_COMMIT' in text
    assert '--array="0-${array_max}%8"' in text
    assert "sbatch" not in text.split('if [[ "$mode" == "dry-run" ]]', maxsplit=1)[0]
