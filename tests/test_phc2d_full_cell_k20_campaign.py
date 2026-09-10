"""Contract tests for the frozen K=20 B/G seed-0 campaign.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

from phc2d.forward import full_cell_k20_campaign as campaign


def _audit(lane: str) -> dict[str, object]:
    return {
        "lane": lane,
        "dataset_path": f"data/{lane}.h5",
        "grid_field": "material_area_fraction_grid_64" if lane == "B" else "material_weight_grid_64",
        "dataset_sha256": f"{lane.lower():0<64}",
        "k_points_sha256": "k" * 64,
        "idx_train_hash": "t" * 64,
        "idx_val_hash": "v" * 64,
        "idx_test_hash": "e" * 64,
    }


def test_task_matrix_contains_exactly_seventeen_seed_zero_dense_tasks(tmp_path: Path) -> None:
    rows = campaign._task_rows(tmp_path, stage="formal", audits={"B": _audit("B"), "G": _audit("G")})
    assert len(rows) == 17
    assert [row["task_index"] for row in rows] == list(range(17))
    assert {row["seed"] for row in rows} == {0}
    assert {row["head"] for row in rows} == {"dense"}
    assert [row["train_lane"] for row in rows[-1:]] == ["J"]
    assert sum(row["train_lane"] == "B" for row in rows) == 8
    assert sum(row["train_lane"] == "G" for row in rows) == 8


def test_freeze_records_shared_k_points_and_test_delay(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(campaign, "_audit_dataset", lambda _root, lane: _audit(lane))
    out = campaign.freeze_campaign(tmp_path / "campaign", repository_root=root)
    payload = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    assert payload["status"] == "frozen"
    assert payload["task_count"] == 17
    assert payload["test_read_before_checkpoint_freeze"] is False
    assert (out / "task_manifest.csv").is_file()
    assert (out / "admission_manifest.csv").is_file()
    audit = json.loads((out / "dataset_audit.json").read_text(encoding="utf-8"))
    assert audit["k_points_shared"] is True


def test_plan_evaluation_accepts_frozen_dual_lane_checkpoint(tmp_path: Path) -> None:
    rows = campaign._task_rows(tmp_path, stage="formal", audits={"B": _audit("B"), "G": _audit("G")})
    campaign._write_csv(tmp_path / "task_manifest.csv", rows)
    for row in rows:
        run = Path(row["run_dir"])
        run.mkdir(parents=True)
        run.joinpath("model.pt").write_bytes(b"checkpoint")
        status = "dual_lane_training_success" if row["train_lane"] == "J" else "formal_training_success"
        run.joinpath("results.json").write_text(
            json.dumps({"status": status, "test_executed": False}), encoding="utf-8"
        )

    out = campaign.plan_evaluation(tmp_path)

    assert out == tmp_path
    assert len(campaign._read_csv(tmp_path / "checkpoint_inventory.csv")) == 17
    assert len(campaign._read_csv(tmp_path / "evaluation_cells_plan.csv")) == 34
