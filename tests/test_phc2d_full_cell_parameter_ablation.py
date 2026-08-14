"""Tests for the Full-cell parameter-ablation freeze and audit workflow.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from phc2d.forward.full_cell_dual_lane import _validate_and_resolve
from phc2d.forward.full_cell_parameter_ablation import (
    SELECTABLE_HEAD_MODELS,
    audit_stage,
    freeze_parameter_ablation,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "research" / "full_cell_forward_parameter_ablation_v1.yaml"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_freeze_emits_complete_ledger_and_bounded_wave(tmp_path: Path) -> None:
    out = freeze_parameter_ablation(CONTRACT, tmp_path / "campaign", root=ROOT)

    ledger = _rows(out / "parameter_ledger.csv")
    full_matrix = _rows(out / "ablation_matrix.csv")
    stage_a = _rows(out / "stage_a_manifest.csv")
    stage_b = _rows(out / "stage_b_wave1_manifest.csv")
    assert len(ledger) == 58
    assert len(full_matrix) == 174
    assert len(stage_a) == 17
    assert len(stage_b) == 51
    assert {row["model"] for row in stage_a} == {
        "mlp",
        "res_mlp_wide",
        "lifted_cnn",
        "fno2d_band_v2",
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
        "dual_lane_band_operator_v1",
    }
    assert all(row["seed"] == "0" for row in stage_b)
    assert all(row["max_epochs"] == "20" for row in stage_b)
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["test_executed"] is False


def test_freeze_rejects_current_value_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    payload["models"]["mlp"]["critical_parameters"]["hidden_dim"]["current"] = 255
    config = tmp_path / "drift.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="current-value drift"):
        freeze_parameter_ablation(config, tmp_path / "campaign", root=ROOT)


def test_dual_lane_short_screen_is_narrowly_admitted() -> None:
    source = yaml.safe_load(
        (ROOT / "configs" / "research" / "full_cell_forward_upgrade_dual_lane_50k_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["experiment"]["evidence_status"] = "single_seed_short_budget_screen"
    source["training"].update(
        {"max_epochs": 20, "epoch_100_continue_gate": False, "early_stopping_patience": 0}
    )

    resolved = _validate_and_resolve(source, seed=0, admission_probe=False, head_kind="dense")
    assert resolved["training"]["max_epochs"] == 20
    with pytest.raises(ValueError, match="seed=0"):
        _validate_and_resolve(source, seed=1, admission_probe=False, head_kind="dense")


def test_audit_stage_fails_closed_on_missing_case(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    (first / "results.json").write_text(
        json.dumps({"status": "full_cell_forward_admission_probe_pass", "test_executed": False}),
        encoding="utf-8",
    )
    (first / "metrics.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "run_dir"])
        writer.writeheader()
        writer.writerow({"case_id": "ok", "run_dir": first})
        writer.writerow({"case_id": "missing", "run_dir": tmp_path / "missing"})

    output = audit_stage(manifest, tmp_path / "audit.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["audited_cases"] == 1
    assert payload["failures"] == [{"case_id": "missing", "reason": "missing_results_or_metrics"}]


def test_selectable_head_override_excludes_baselines() -> None:
    assert SELECTABLE_HEAD_MODELS == {
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
    }
    assert not ({"mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2"} & SELECTABLE_HEAD_MODELS)
