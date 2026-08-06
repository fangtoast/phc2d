"""Tests for full-cell cloud versus WSL replay auditing.

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

from phc2d.forward.full_cell_replay_audit import audit_full_cell_replay


def _write_final_test(path: Path, *, delta: float = 0.0) -> None:
    path.mkdir()
    (path / "results.json").write_text(json.dumps({"status": "final_test_evaluation_success"}), encoding="utf-8")
    fields = ["lane", "representation", "model", "resolution", "seed", "mre_pct", "mae", "rmse"]
    with (path / "test_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for lane, representation in (("binary", "binary"), ("graded", "continuous"), ("graded", "t06")):
            for seed in range(3):
                writer.writerow(
                    {
                        "lane": lane,
                        "representation": representation,
                        "model": "fno2d_band_v2",
                        "resolution": 64,
                        "seed": seed,
                        "mre_pct": 1.0 + delta,
                        "mae": 0.1 + delta,
                        "rmse": 0.2 + delta,
                    }
                )
    for lane, representation in (("binary", "binary"), ("graded", "continuous"), ("graded", "t06")):
        np.savez_compressed(
            path / f"predictions_test_{lane}_{representation}.npz",
            indices=np.asarray([0, 1]),
            sample_id=np.asarray(["a", "b"]),
            target_Wh=np.ones((2, 2, 2), dtype=np.float32),
            prediction_seed0=np.ones((2, 2, 2), dtype=np.float32) + delta,
            prediction_seed1=np.ones((2, 2, 2), dtype=np.float32) + delta,
            prediction_seed2=np.ones((2, 2, 2), dtype=np.float32) + delta,
        )


def test_replay_audit_accepts_close_outputs(tmp_path: Path) -> None:
    cluster = tmp_path / "cluster"
    wsl = tmp_path / "wsl"
    _write_final_test(cluster)
    _write_final_test(wsl, delta=1.0e-7)

    output = audit_full_cell_replay(cluster, wsl_dir=wsl, output_dir=tmp_path / "audit")

    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert results["status"] == "wsl_checkpoint_replay_acceptance_success"
    assert results["metric_row_count"] == 9
    assert results["prediction_artifact_count"] == 3


def test_replay_audit_rejects_prediction_mismatch(tmp_path: Path) -> None:
    cluster = tmp_path / "cluster"
    wsl = tmp_path / "wsl"
    _write_final_test(cluster)
    _write_final_test(wsl, delta=0.1)

    with pytest.raises(RuntimeError, match="metric replay mismatch"):
        audit_full_cell_replay(cluster, wsl_dir=wsl, output_dir=tmp_path / "audit")
