"""Tests for formal full-cell forward wave aggregation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phc2d.forward.full_cell_wave import MODELS, aggregate_full_cell_forward_wave


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_aggregate_full_cell_forward_wave_freezes_validation_only_ranking(tmp_path: Path) -> None:
    manifest_rows = []
    for model_index, model in enumerate(MODELS):
        for seed in (0, 1, 2):
            run = tmp_path / "runs" / f"{model}_seed{seed}"
            (run / "figures").mkdir(parents=True)
            value = 4.0 - model_index + seed * 0.01
            (run / "run.log").write_text("terminal_status=formal_training_success\n", encoding="utf-8")
            (run / "config.resolved.yaml").write_text("training: {}\n", encoding="utf-8")
            (run / "model.pt").write_bytes(b"checkpoint")
            (run / "figures" / "training_curve.png").write_bytes(b"png")
            (run / "figures" / "training_curve.pdf").write_bytes(b"pdf")
            (run / "results.json").write_text(
                json.dumps(
                    {
                        "status": "formal_training_success",
                        "test_executed": False,
                        "selection_split": "validation",
                        "checkpoint_path": str(run / "model.pt"),
                        "checkpoint_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            (run / "metrics.json").write_text(
                json.dumps(
                    {
                        "best_validation_bands_2_10_mre_pct": value,
                        "best_epoch": 1,
                        "stop_epoch": 2,
                        "train_time_s": 1.0,
                        "wall_time_total_s": 2.0,
                        "peak_gpu_memory_mb": 3.0,
                    }
                ),
                encoding="utf-8",
            )
            (run / "run.meta.json").write_text(
                json.dumps(
                    {
                        "dataset_sha256": "d" * 64,
                        "full_dataset_final_fit": False,
                        "idx_train_hash": "1" * 64,
                        "idx_val_hash": "2" * 64,
                        "idx_test_hash": "3" * 64,
                        "config_path": "configs/research/wave.yaml",
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(
                run / "history.csv",
                [
                    {"epoch": 1, "val_bands_2_10_mre_pct": value + 0.1},
                    {"epoch": 2, "val_bands_2_10_mre_pct": value},
                ],
            )
            manifest_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "job_id": 100 + len(manifest_rows),
                    "run_dir": str(run.relative_to(tmp_path)),
                    "mode": "train",
                }
            )
    manifest = tmp_path / "submission_manifest.csv"
    _write_csv(manifest, manifest_rows)
    out = aggregate_full_cell_forward_wave(
        manifest,
        output_dir="runs/wave_aggregation",
        repository_root=tmp_path,
    )
    meta = json.loads((out / "wave.meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "formal_wave_aggregation_success"
    assert meta["completed_runs"] == 12
    assert meta["validation_winner"] == "fno2d_band_v2"
    assert meta["test_executed"] is False
    for name in (
        "run_manifest.csv",
        "seed_summary.csv",
        "leaderboard.csv",
        "figures/training_curves_by_model.png",
        "figures/training_curves_by_model.pdf",
        "figures/validation_comparison.png",
        "figures/validation_comparison.pdf",
    ):
        assert (out / name).is_file()


def test_aggregate_final_fit_wave_never_creates_validation_winner(tmp_path: Path) -> None:
    manifest_rows = []
    for model_index, model in enumerate(MODELS):
        for seed in (0, 1, 2):
            run = tmp_path / "runs" / f"final_fit_{model}_seed{seed}"
            (run / "figures").mkdir(parents=True)
            value = 4.0 - model_index + seed * 0.01
            (run / "run.log").write_text("terminal_status=formal_training_success\n", encoding="utf-8")
            (run / "config.resolved.yaml").write_text("training: {train_size: 50000}\n", encoding="utf-8")
            (run / "model.pt").write_bytes(b"checkpoint")
            (run / "figures" / "training_curve.png").write_bytes(b"png")
            (run / "figures" / "training_curve.pdf").write_bytes(b"pdf")
            (run / "results.json").write_text(
                json.dumps(
                    {
                        "status": "formal_training_success",
                        "test_executed": False,
                        "selection_split": "none_final_fit",
                        "checkpoint_path": str(run / "model.pt"),
                        "checkpoint_sha256": "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            (run / "metrics.json").write_text(
                json.dumps(
                    {
                        "best_training_overlap_monitor_mre_pct": value,
                        "best_epoch": 2,
                        "stop_epoch": 2,
                        "train_time_s": 1.0,
                        "wall_time_total_s": 2.0,
                        "peak_gpu_memory_mb": 3.0,
                    }
                ),
                encoding="utf-8",
            )
            (run / "run.meta.json").write_text(
                json.dumps(
                    {
                        "dataset_sha256": "d" * 64,
                        "idx_train_hash": "1" * 64,
                        "idx_val_hash": "2" * 64,
                        "idx_test_hash": "3" * 64,
                        "config_path": "configs/research/wave.yaml",
                        "full_dataset_final_fit": True,
                    }
                ),
                encoding="utf-8",
            )
            _write_csv(
                run / "history.csv",
                [
                    {"epoch": 1, "val_bands_2_10_mre_pct": value + 0.1},
                    {"epoch": 2, "val_bands_2_10_mre_pct": value},
                ],
            )
            manifest_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "job_id": 200 + len(manifest_rows),
                    "run_dir": str(run.relative_to(tmp_path)),
                    "mode": "train",
                }
            )
    manifest = tmp_path / "final_fit_submission_manifest.csv"
    _write_csv(manifest, manifest_rows)
    out = aggregate_full_cell_forward_wave(
        manifest,
        output_dir="runs/final_fit_wave_aggregation",
        repository_root=tmp_path,
    )
    meta = json.loads((out / "wave.meta.json").read_text(encoding="utf-8"))
    assert meta["selection_split"] == "none_final_fit"
    assert meta["monitor_only_no_model_selection"] is True
    assert meta["validation_winner"] is None
    assert not (out / "leaderboard.csv").exists()
    for name in (
        "training_monitor_summary.csv",
        "figures/training_monitor_comparison.png",
        "figures/training_monitor_comparison.pdf",
    ):
        assert (out / name).is_file()
