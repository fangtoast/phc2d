"""Tests for validation-only full-cell main winner freezing.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phc2d.forward.full_cell_campaign import freeze_full_cell_main_winners
from phc2d.forward.full_cell_cross_eval import WAVE_LABELS
from phc2d.forward.full_cell_wave import MODELS


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_freeze_main_winners_requires_all_84_runs(tmp_path: Path) -> None:
    args = []
    for wave_index, label in enumerate(WAVE_LABELS):
        directory = tmp_path / label
        directory.mkdir()
        (directory / "wave.meta.json").write_text(
            json.dumps(
                {
                    "status": "formal_wave_aggregation_success",
                    "completed_runs": 12,
                    "test_executed": False,
                    "test_driven_selection": False,
                    "config_path": f"configs/{label}.yaml",
                }
            ),
            encoding="utf-8",
        )
        _write_csv(
            directory / "run_manifest.csv",
            [
                {"model": model, "seed": seed, "status": "formal_training_success"}
                for model in MODELS
                for seed in (0, 1, 2)
            ],
        )
        _write_csv(
            directory / "leaderboard.csv",
            [
                {
                    "model": model,
                    "validation_mre_mean_pct": 10.0 - model_index - wave_index * 0.1,
                    "validation_mre_std_pct": 0.01,
                    "success_count": 3,
                }
                for model_index, model in enumerate(MODELS)
            ],
        )
        args.append(f"{label}={directory}")
    out = freeze_full_cell_main_winners(args, output_dir="campaign", repository_root=tmp_path)
    meta = json.loads((out / "campaign.meta.json").read_text(encoding="utf-8"))
    selections = list(csv.DictReader((out / "winner_selection.csv").open(encoding="utf-8")))
    assert meta["status"] == "main_84_training_complete_validation_winners_frozen"
    assert meta["completed_runs"] == 84
    assert len(selections) == 3
    assert {row["lane"] for row in selections} == {"binary", "graded"}
    assert all(row["model"] == "fno2d_band_v2" for row in selections)
