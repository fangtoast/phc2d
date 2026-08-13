"""Tests for the frozen Full-Wh upgrade 50k confirmation matrix.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from phc2d.forward.full_cell_upgrade_50k import WAVES, build_50k_manifest
from phc2d.forward.full_cell import _validate_and_resolve as resolve_single
from phc2d.forward.full_cell_dual_lane import _validate_and_resolve as resolve_dual


ROOT = Path(__file__).resolve().parents[1]


def _locks(root: Path, *, not_admitted: str | None = None) -> tuple[Path, Path]:
    models = (
        "mlp_mixer_band_v1",
        "res_mlp_gated_v2",
        "convnext_band_v1",
        "tfno2d_band_v3",
    )
    selections = [
        {"model": model, "lane": lane, "selected_head": "dense"}
        for model in models
        for lane in ("B", "G")
    ] + [{"model": "dual_lane_band_operator_v1", "lane": "J", "selected_head": "dense"}]
    head = root / "head.json"
    head.write_text(
        json.dumps({"status": "head_selection_frozen", "test_read_during_selection": False, "selections": selections}),
        encoding="utf-8",
    )
    decisions = [
        {"model": model, "status": "not_admitted" if model == not_admitted else "admitted"}
        for model in (*models, "dual_lane_band_operator_v1")
    ]
    admission = root / "admission.json"
    admission.write_text(
        json.dumps(
            {
                "status": "50k_admission_frozen",
                "test_influenced_head_selection": False,
                "decisions": decisions,
            }
        ),
        encoding="utf-8",
    )
    return head, admission


def test_manifest_preserves_all_27_rows_and_preregistered_waves(tmp_path: Path) -> None:
    head, admission = _locks(tmp_path)
    out = build_50k_manifest(head, admission, output_dir="campaign", repository_root=tmp_path)
    rows = list(csv.DictReader((out / "task_manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 27
    assert {row["wave"] for row in rows} == set(WAVES)
    assert [sum(row["wave"] == wave for row in rows) for wave in WAVES] == [8, 8, 8, 3]
    assert all(row["head"] == "dense" and row["action"] == "RUN" for row in rows)


def test_manifest_keeps_not_admitted_slots_without_reordering(tmp_path: Path) -> None:
    head, admission = _locks(tmp_path, not_admitted="convnext_band_v1")
    out = build_50k_manifest(head, admission, output_dir="campaign", repository_root=tmp_path)
    rows = list(csv.DictReader((out / "task_manifest.csv").open(encoding="utf-8")))
    skipped = [row for row in rows if row["action"] == "SKIP_NOT_ADMITTED"]
    assert len(skipped) == 6
    assert [int(row["task_index"]) for row in rows] == list(range(27))


def test_50k_configs_freeze_data_training_and_epoch_gate() -> None:
    configs = [
        yaml.safe_load((ROOT / "configs/research/full_cell_forward_upgrade_binary_50k_v1.yaml").read_text(encoding="utf-8")),
        yaml.safe_load((ROOT / "configs/research/full_cell_forward_upgrade_graded_50k_v1.yaml").read_text(encoding="utf-8")),
        yaml.safe_load((ROOT / "configs/research/full_cell_forward_upgrade_dual_lane_50k_v1.yaml").read_text(encoding="utf-8")),
    ]
    for config in configs:
        training = config["training"]
        assert training["train_size"] == 40000
        assert training["max_epochs"] == 200
        assert training["epoch_100_continue_gate"] is True
        assert training["continue_gate_min_relative_improvement"] == 0.005
        assert training["early_stopping_patience"] == 15
        assert config["runtime"]["precision"] == "fp32"
        assert config["runtime"]["amp_enabled"] is False
    for config in configs[:2]:
        for model in config["models"]:
            resolved = resolve_single(
                config,
                model_name=model,
                seed=0,
                admission_probe=False,
                head_kind_override="dense",
            )
            assert resolved["training"]["epoch_100_continue_gate"] is True
    dual = resolve_dual(configs[2], seed=0, admission_probe=False, head_kind="dense")
    assert dual["training"]["epoch_100_continue_gate"] is True
