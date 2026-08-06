"""Tests for full-cell resolution-ablation acceptance.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phc2d.forward.full_cell_ablation import aggregate_full_cell_resolution_ablation


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_ablation_acceptance_freezes_resolution_without_test(tmp_path: Path) -> None:
    selections = [
        {"lane": "binary", "representation": "binary", "config": "binary.yaml", "model": "fno2d_band_v2", "threshold": "", "validation_mre_mean_pct": 0.8, "validation_mre_std_pct": 0.01},
        {"lane": "graded", "representation": "continuous", "config": "continuous.yaml", "model": "fno2d_band_v2", "threshold": "", "validation_mre_mean_pct": 0.4, "validation_mre_std_pct": 0.01},
        {"lane": "graded", "representation": "t05", "config": "threshold.yaml", "model": "fno2d_band_v2", "threshold": 0.5, "validation_mre_mean_pct": 0.6, "validation_mre_std_pct": 0.01},
    ]
    selection_path = tmp_path / "winner_selection.csv"
    _write_csv(selection_path, selections)
    manifest_rows = []
    for selection in selections:
        for resolution in (32, 128):
            for seed in (0, 1, 2):
                run = tmp_path / "runs" / f"{selection['representation']}_r{resolution}_s{seed}"
                (run / "figures").mkdir(parents=True)
                for name in ("run.log", "config.resolved.yaml", "history.csv"):
                    (run / name).write_text("ok\n", encoding="utf-8")
                (run / "model.pt").write_bytes(b"checkpoint")
                (run / "figures" / "training_curve.png").write_bytes(b"png")
                (run / "figures" / "training_curve.pdf").write_bytes(b"pdf")
                (run / "run.meta.json").write_text("{}", encoding="utf-8")
                (run / "results.json").write_text(
                    json.dumps({"status": "formal_training_success", "test_executed": False, "checkpoint_sha256": "a" * 64}),
                    encoding="utf-8",
                )
                value = float(selection["validation_mre_mean_pct"]) + (0.2 if resolution == 32 else 0.3) + seed * 0.01
                (run / "metrics.json").write_text(
                    json.dumps({"best_validation_bands_2_10_mre_pct": value}),
                    encoding="utf-8",
                )
                manifest_rows.append(
                    {
                        "lane": selection["lane"],
                        "representation": selection["representation"],
                        "config": selection["config"],
                        "model": selection["model"],
                        "threshold": selection["threshold"],
                        "resolution": resolution,
                        "seed": seed,
                        "job_id": len(manifest_rows) + 1,
                        "run_dir": str(run.relative_to(tmp_path)),
                        "mode": "train",
                    }
                )
    manifest = tmp_path / "ablation_manifest.csv"
    _write_csv(manifest, manifest_rows)
    main_manifest = tmp_path / "main_matrix_manifest.csv"
    _write_csv(main_manifest, [{"run": index} for index in range(84)])
    out = aggregate_full_cell_resolution_ablation(
        selection_path,
        manifest,
        main_matrix_manifest=main_manifest,
        output_dir="acceptance",
        repository_root=tmp_path,
    )
    meta = json.loads((out / "acceptance.meta.json").read_text(encoding="utf-8"))
    final = list(csv.DictReader((out / "final_resolution_selection.csv").open(encoding="utf-8")))
    assert meta["status"] == "all_102_forward_training_runs_complete_final_resolutions_frozen"
    assert meta["total_training_runs"] == 102
    assert len(final) == 3
    assert all(int(row["resolution"]) == 64 for row in final)
    assert all(row["test_driven_selection"] == "False" for row in final)
