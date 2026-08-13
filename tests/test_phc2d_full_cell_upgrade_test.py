"""Tests for frozen-head Full-Wh held-out evaluation orchestration.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from phc2d.forward.full_cell_upgrade_test import (
    DIRECT_BASELINES,
    build_test_manifest,
    finalize_test_campaign,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_inputs(root: Path) -> tuple[Path, Path, Path]:
    selections: list[dict[str, object]] = []
    for model in tuple(DIRECT_BASELINES)[:-1]:
        for lane in ("B", "G"):
            selected_runs = []
            for seed in (0, 1, 2):
                run = root / "upgrade" / model / lane / str(seed)
                digest = _checkpoint(run / "model.pt", f"{model}-{lane}-{seed}")
                selected_runs.append({"seed": seed, "run_dir": str(run.relative_to(root)), "checkpoint_sha256": digest})
            selections.append({"model": model, "lane": lane, "selected_head": "dense", "selected_runs": selected_runs})
    model = "dual_lane_band_operator_v1"
    selected_runs = []
    for seed in (0, 1, 2):
        run = root / "upgrade" / model / "J" / str(seed)
        digest = _checkpoint(run / "model.pt", f"{model}-J-{seed}")
        selected_runs.append({"seed": seed, "run_dir": str(run.relative_to(root)), "checkpoint_sha256": digest})
    selections.append({"model": model, "lane": "J", "selected_head": "k_conditioned", "selected_runs": selected_runs})
    lock = root / "head_selection.lock.json"
    lock.write_text(
        json.dumps({"status": "head_selection_frozen", "test_read_during_selection": False, "selections": selections}),
        encoding="utf-8",
    )
    manifests = []
    for lane in ("B", "G"):
        rows = []
        for baseline_model in ("mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2"):
            for seed in (0, 1, 2):
                checkpoint = root / "baseline" / lane / baseline_model / str(seed) / "model.pt"
                digest = _checkpoint(checkpoint, f"{lane}-{baseline_model}-{seed}")
                rows.append(
                    {
                        "model": baseline_model,
                        "seed": seed,
                        "status": "formal_training_success",
                        "run_dir": str(checkpoint.parent.relative_to(root)),
                        "checkpoint_path": str(checkpoint.relative_to(root)),
                        "checkpoint_sha256": digest,
                        "test_executed": False,
                    }
                )
        manifest = root / f"baseline_{lane}.csv"
        _write_csv(manifest, rows)
        manifests.append(manifest)
    return lock, manifests[0], manifests[1]


def test_build_manifest_freezes_51_hash_verified_tasks(tmp_path: Path) -> None:
    lock, binary, graded = _freeze_inputs(tmp_path)
    campaign = build_test_manifest(
        lock,
        baseline_binary_manifest=binary,
        baseline_graded_manifest=graded,
        output_dir="campaign",
        repository_root=tmp_path,
    )
    rows = list(csv.DictReader((campaign / "test_manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 51
    assert sum(row["source_kind"] == "upgrade" for row in rows) == 27
    assert sum(row["source_kind"] == "baseline" for row in rows) == 24
    assert json.loads((campaign / "selection_input.json").read_text())["evaluation_split"] == "test"


def _materialize_result(root: Path, task: dict[str, str]) -> None:
    result_dir = root / task["result_dir"]
    result_dir.mkdir(parents=True)
    lanes = ("B", "G") if task["lane"] == "J" else (task["lane"],)
    evaluations = []
    sample_rows = []
    for lane in lanes:
        baseline = task["source_kind"] == "baseline"
        # Only Mixer clears its direct gate: 10% better on B and 2% worse on G.
        if baseline:
            value = 1.0
        elif task["model"] == "mlp_mixer_band_v1":
            value = 0.90 if lane == "B" else 1.02
        elif task["model"] == "dual_lane_band_operator_v1":
            value = 1.03
        else:
            value = 1.01
        evaluations.append(
            {
                "lane": lane,
                "test_bands_2_10_mre_pct": value,
                "test_full_matrix_mre_pct": value,
                "test_mae": value,
                "test_rmse": value,
                "test_ordering_violation_rate": 0.0,
                "n_test": 4,
                "eval_time_s": 0.1,
                "inference_time_per_sample_ms": 25.0,
                "throughput_samples_per_s": 40.0,
            }
        )
        for order in range(4):
            sample_rows.append(
                {
                    "lane": lane,
                    "sample_order": order,
                    "dataset_index": order,
                    "bands_2_10_mre_pct": value,
                    "full_matrix_mre_pct": value,
                    "ordering_violation_rate": 0.0,
                }
            )
    (result_dir / "results.json").write_text(
        json.dumps({"status": "frozen_checkpoint_test_success", "checkpoint_sha256": task["checkpoint_sha256"]}),
        encoding="utf-8",
    )
    (result_dir / "metrics.json").write_text(
        json.dumps({"evaluations": evaluations, "parameter_count": 2_500_000, "peak_gpu_memory_mb": 900.0}),
        encoding="utf-8",
    )
    _write_csv(result_dir / "per_sample_metrics.csv", sample_rows)
    _write_csv(result_dir / "per_band_metrics.csv", [{"lane": lanes[0], "band_one_based": 1, "mre_pct": 1.0}])
    for name in ("history.csv", "config.resolved.yaml", "run.meta.json", "run.log"):
        (result_dir / name).write_text("synthetic\n", encoding="utf-8")


def test_finalize_freezes_admission_and_paired_ci(tmp_path: Path) -> None:
    lock, binary, graded = _freeze_inputs(tmp_path)
    campaign = build_test_manifest(
        lock,
        baseline_binary_manifest=binary,
        baseline_graded_manifest=graded,
        output_dir="campaign",
        repository_root=tmp_path,
    )
    tasks = list(csv.DictReader((campaign / "test_manifest.csv").open(encoding="utf-8")))
    for task in tasks:
        _materialize_result(tmp_path, task)
    finalize_test_campaign(campaign, repository_root=tmp_path)
    admission = json.loads((campaign / "50k_admission.lock.json").read_text(encoding="utf-8"))
    statuses = {item["model"]: item["status"] for item in admission["decisions"]}
    assert statuses["mlp_mixer_band_v1"] == "admitted"
    assert statuses["dual_lane_band_operator_v1"] == "admitted"
    assert statuses["res_mlp_gated_v2"] == "not_admitted"
    assert len(list(csv.DictReader((campaign / "test_runs.csv").open(encoding="utf-8")))) == 54
    assert len(list(csv.DictReader((campaign / "paired_bootstrap_ci.csv").open(encoding="utf-8")))) == 10
