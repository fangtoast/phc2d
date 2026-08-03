"""Tests for inverse sampled band-curve plotting.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import plot_inverse_test625_band_curves as plotter


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_inverse_band_curve_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plotter, "repo_root", lambda: tmp_path)
    summary = tmp_path / "runs/20260706/summary"
    (summary / "summary_tables").mkdir(parents=True)
    shape = (5, 4, 3)
    base = np.linspace(0.2, 0.8, num=np.prod(shape), dtype=np.float32).reshape(shape)
    np.save(summary / "target_bands.npy", base)
    np.save(summary / "surrogate_pred_bands.npy", base * 1.02)
    np.save(summary / "mpb_replay_bands.npy", base * 1.04)
    rows = [
        {
            "target_index": idx,
            "best_candidate_selected_index": 1,
            "best_of_k_replay_mre_bands_2_10_pct": 4.0 + idx,
            "best_of_k_surrogate_mre_bands_2_10_pct": 2.0 + idx,
            "surrogate_replay_gap_bands_2_10_pct": 2.0,
        }
        for idx in range(shape[0])
    ]
    _write_csv(
        summary / "summary_tables/per_target_replay_metrics.csv",
        rows,
    )
    dataset = tmp_path / "data/fake.npz"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dataset, rho=np.zeros((shape[0], 36), dtype=np.float32))
    candidates = tmp_path / "runs/20260706/sample/candidates.json"
    candidates.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "target_index": idx,
                        "candidates": [
                            np.zeros(36, dtype=float).tolist(),
                            np.ones(36, dtype=float).tolist(),
                            np.eye(6, dtype=float).reshape(-1).tolist(),
                        ],
                        "selected_indices": [1],
                        "selected_surrogate_predictions": [(base[idx] * 1.02).reshape(-1).tolist()],
                    }
                    for idx in range(shape[0])
                ]
            }
        ),
        encoding="utf-8",
    )
    (summary / "source_shard_manifest.json").write_text(
        json.dumps(
            {
                "dataset_path": "data/fake.npz",
                "candidates_path": "runs/20260706/sample/candidates.json",
            }
        ),
        encoding="utf-8",
    )
    (summary / "artifact_index.json").write_text('{"artifacts": []}\n', encoding="utf-8")

    rc = plotter.main(
        [
            "--summary-run",
            str(summary),
            "--n-samples",
            "3",
            "--seed",
            "7",
            "--output-subdir",
            "band_curve_test",
        ]
    )

    assert rc == 0
    out_dir = summary / "figures/band_curve_test"
    assert (out_dir / "band_curve_sample100_overview.png").exists()
    assert (out_dir / "band_curve_sample100_overview.svg").exists()
    assert (out_dir / "structure_band_gallery_sample100.png").exists()
    assert (out_dir / "structure_band_gallery_sample100.svg").exists()
    assert len(list((out_dir / "individual").glob("*.png"))) == 3
    with (out_dir / "band_curve_sample100_index.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    artifact_index = json.loads((summary / "artifact_index.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in artifact_index["artifacts"]}
    assert "runs/20260706/summary/figures/band_curve_test/band_curve_sample100_overview.png" in paths
    assert "runs/20260706/summary/figures/band_curve_test/structure_band_gallery_sample100.png" in paths

    rc = plotter.main(
        [
            "--summary-run",
            str(summary),
            "--n-samples",
            "2",
            "--selection",
            "best",
            "--output-subdir",
            "best_worst_test",
            "--output-prefix",
            "best2",
            "--individual-subdir",
            "best2_individual",
        ]
    )
    assert rc == 0
    best_dir = summary / "figures/best_worst_test"
    assert (best_dir / "structure_band_gallery_best2.png").exists()
    with (best_dir / "best2_index.csv").open(newline="", encoding="utf-8") as fh:
        best_rows = list(csv.DictReader(fh))
    assert [row["target_index"] for row in best_rows] == ["0", "1"]
