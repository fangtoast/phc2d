"""Tests for inverse 625-test replay shard planning and summarization.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import prepare_inverse_test625_replay_shards as prepare
from scripts import summarize_inverse_test625_replay_shards as summarize


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ",".join(header) + "\n"
    text += "".join(",".join(str(item) for item in row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def test_prepare_inverse_test625_replay_shards_exact_idx_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(prepare, "repo_root", lambda: tmp_path)
    dataset = tmp_path / "data/query_dataset_v2_pilot_10000_mpb.npz"
    dataset.parent.mkdir(parents=True)
    np.savez(dataset, idx_test=np.arange(10, 15, dtype=np.int64))
    candidates = tmp_path / "sample/candidates.json"
    _write_json(candidates, {"version": 1, "targets": [{"target_index": int(index)} for index in range(10, 15)]})
    base_config = tmp_path / "configs/base.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text(
        yaml.safe_dump(
            {
                "experiment": {"experiment_id": "base"},
                "dataset": {"path": "data/query_dataset_v2_pilot_10000_mpb.npz"},
                "replay": {"backend": "real_mpb", "replay_top_m": 5},
                "output": {"runs_dir": "runs"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs/20260706/test625_plan"

    rc = prepare.main(
        [
            "--candidates",
            str(candidates),
            "--dataset",
            str(dataset),
            "--base-config",
            str(base_config),
            "--output-root",
            str(output_root),
            "--shard-size",
            "2",
            "--timestamp",
            "20260706123456",
        ]
    )

    assert rc == 0
    manifest = json.loads((output_root / "shard_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_count"] == 5
    assert manifest["shard_count"] == 3
    assert manifest["replay_backend"] == "real_mpb"
    shard_cfg = yaml.safe_load((output_root / "configs/shard_00.yaml").read_text(encoding="utf-8"))
    assert shard_cfg["replay"]["target_indices_path"] == "runs/20260706/test625_plan/target_indices/shard_00_target_indices.json"
    assert shard_cfg["replay"]["n_targets"] == 2
    assert shard_cfg["replay"]["require_real_mpb"] is True


def test_prepare_inverse_test625_replay_shards_rejects_non_test_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(prepare, "repo_root", lambda: tmp_path)
    dataset = tmp_path / "data/query_dataset_v2_pilot_10000_mpb.npz"
    dataset.parent.mkdir(parents=True)
    np.savez(dataset, idx_test=np.array([10, 11], dtype=np.int64))
    candidates = tmp_path / "sample/candidates.json"
    _write_json(candidates, {"targets": [{"target_index": 10}, {"target_index": 12}]})
    base_config = tmp_path / "configs/base.yaml"
    base_config.parent.mkdir(parents=True)
    base_config.write_text("replay: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly equal"):
        prepare.main(
            [
                "--candidates",
                str(candidates),
                "--dataset",
                str(dataset),
                "--base-config",
                str(base_config),
                "--output-root",
                str(tmp_path / "plan"),
            ]
        )


def _write_fake_shard_run(run_dir: Path, shard_id: int, target_indices: list[int], *, nk: int = 3, nb: int = 4) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run.log").write_text("ok\n", encoding="utf-8")
    _write_json(run_dir / "run.meta.json", {"run": "fake"})
    (run_dir / "config.resolved.yaml").write_text("replay:\n  backend: real_mpb\n", encoding="utf-8")
    _write_json(run_dir / "results.json", {"ok": True})
    _write_json(
        run_dir / "metrics.json",
        {
            "replay_backend": "real_mpb",
            "validation_boundary": "formal_inverse_candidate_mpb_replay",
            "targets": len(target_indices),
        },
    )
    rows = [
        [
            "replay",
            f"target_{index}",
            index,
            5,
            2.0 + 0.1 * offset + shard_id,
            1.0 + 0.1 * offset,
            1.0 + shard_id,
            10.0,
        ]
        for offset, index in enumerate(target_indices)
    ]
    header = [
        "stage",
        "target_id",
        "target_index",
        "candidate_count_selected",
        "best_of_k_replay_mre_bands_2_10_pct",
        "best_of_k_surrogate_mre_bands_2_10_pct",
        "surrogate_replay_gap_bands_2_10_pct",
        "mpb_replay_time_s",
    ]
    _write_csv(run_dir / "history.csv", header, rows)
    _write_csv(run_dir / "per_target_replay_metrics.csv", header, rows)
    _write_csv(run_dir / "per_band_replay_metrics.csv", ["band_one_based", "mpb_replay_mre_pct"], [[1, 2.0]])
    band_rows: list[list[object]] = []
    for index in target_indices:
        for band in range(1, nb + 1):
            band_rows.append([f"target_{index}", index, band, 2.0 + band, 1.0 + band, 1.0])
    _write_csv(
        run_dir / "per_target_band_replay_metrics.csv",
        ["target_id", "target_index", "band_one_based", "mpb_replay_mre_pct", "surrogate_mre_pct", "surrogate_replay_gap_pct"],
        band_rows,
    )
    (run_dir / "replay_candidates.jsonl").write_text('{"target_index": 1}\n', encoding="utf-8")
    shape = (len(target_indices), nk, nb)
    np.save(run_dir / "target_bands.npy", np.ones(shape, dtype=np.float32))
    np.save(run_dir / "surrogate_pred_bands.npy", np.ones(shape, dtype=np.float32) * 1.1)
    np.save(run_dir / "mpb_replay_bands.npy", np.ones(shape, dtype=np.float32) * 1.2)


def test_summarize_inverse_test625_replay_shards_complete(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(summarize, "repo_root", lambda: tmp_path)
    shard_root = tmp_path / "runs/20260706/plan"
    shard_root.mkdir(parents=True)
    shards = []
    for shard_id, target_indices in enumerate(([10, 11], [12, 13, 14])):
        run_dir = tmp_path / f"runs/20260706/shard_{shard_id:02d}"
        _write_fake_shard_run(run_dir, shard_id, list(target_indices))
        shards.append({"shard_id": shard_id, "shard_name": f"shard_{shard_id:02d}", "run_dir": f"runs/20260706/shard_{shard_id:02d}"})
    _write_json(
        shard_root / "shard_manifest.json",
        {"version": 1, "experiment_id": "inverse_mpb_replay_mixture_v2_10000_test625", "shards": shards},
    )
    output_dir = tmp_path / "runs/20260706/summary"
    report_path = tmp_path / "docs/results/report.md"

    rc = summarize.main(
        [
            "--shard-root",
            str(shard_root),
            "--output-dir",
            str(output_dir),
            "--report-path",
            str(report_path),
            "--expected-target-count",
            "5",
            "--expected-nk",
            "3",
            "--expected-nb",
            "4",
            "--bootstrap-repeats",
            "20",
            "--timestamp",
            "20260706123500",
        ]
    )

    assert rc == 0
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "complete"
    assert metrics["target_count_processed"] == 5
    assert metrics["band_array_shape"] == [5, 3, 4]
    assert (output_dir / "summary_tables/worst_20_targets.csv").exists()
    assert "Evidence Boundary" in report_path.read_text(encoding="utf-8")
