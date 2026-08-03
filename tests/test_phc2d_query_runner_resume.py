"""Tests for resumable query benchmark execution.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from phc2d.data import (
    ForwardDatasetBundle,
    KPathInfo,
    build_physical_query_masks,
    save_forward_dataset,
)
from phc2d.query.runner import (
    _latest_query_epoch_checkpoint,
    _protocol_lookup,
    run_query_benchmark,
)


def _write_tiny_query_npz(path: Path) -> None:
    rng = np.random.default_rng(42)
    rho = rng.random((6, 36), dtype=np.float32)
    weights = rng.normal(size=(36, 12), scale=0.01).astype(np.float32)
    wh = (rho @ weights).reshape(6, 4, 3).astype(np.float32)
    save_forward_dataset(
        path,
        ForwardDatasetBundle(
            rho=rho,
            Wh=wh,
            idx_train=np.array([0, 1, 2, 3], dtype=np.int64),
            idx_val=np.array([4], dtype=np.int64),
            idx_test=np.array([5], dtype=np.int64),
            config_yaml="k_path:\n  n_total: 4\n",
        ),
    )


def _write_query_config(
    path: Path,
    dataset: Path,
    runs_dir: Path,
    *,
    run_dir: Path | None = None,
    epochs: int = 1,
) -> None:
    run_dir_lines = [] if run_dir is None else [f"  run_dir: {run_dir.as_posix()}"]
    path.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_query_resume",
                "dataset:",
                f"  path: {dataset.as_posix()}",
                "query:",
                "  requires_physical_k_coordinates: false",
                "  allow_index_query: true",
                "protocols:",
                "  - unseen_structure_all_k",
                "model:",
                "  candidates:",
                "    - mlp_query",
                "  params:",
                "    hidden_dim: 8",
                "    depth: 1",
                "training:",
                f"  epochs: {int(epochs)}",
                "  checkpoint_epochs: [1]",
                "  batch_size: 4",
                "  learning_rate: 0.001",
                "  seed: 0",
                "  device: cpu",
                "limits:",
                "  max_train_structures: 2",
                "  max_eval_structures: 1",
                "  max_eval_queries: 4",
                "output:",
                f"  runs_dir: {runs_dir.as_posix()}",
                "  write_query_report: false",
                "  write_dataset_audit_report: false",
                *run_dir_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _history_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_latest_query_epoch_checkpoint_ignores_nonmatching_files(tmp_path: Path) -> None:
    cases = tmp_path / "runs" / "case_run" / "cases"
    cases.mkdir(parents=True)
    (cases / "case_a_checkpoint_epoch_1.pt").write_text("x", encoding="utf-8")
    (cases / "case_a_checkpoint_epoch_3.pt").write_text("x", encoding="utf-8")
    (cases / "case_a_checkpoint_epoch_bad.pt").write_text("x", encoding="utf-8")
    (cases / "case_ab_checkpoint_epoch_9.pt").write_text("x", encoding="utf-8")

    latest = _latest_query_epoch_checkpoint(tmp_path / "runs" / "case_run", "case_a")

    assert latest is not None
    assert latest[0] == 3
    assert latest[1].name == "case_a_checkpoint_epoch_3.pt"


def test_physical_query_protocols_resolve_by_reported_names() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.2, 0.1], [0.2, 0.2], [0.1, 0.1], [0.0, 0.0]],
        dtype=np.float32,
    )
    k_path = KPathInfo(
        coordinates=coordinates,
        source="unit",
        is_physical=True,
        has_segment_metadata=True,
        repeated_endpoint_indices=(6,),
        segment_id=np.asarray([0, 0, 0, 1, 1, 2, 2], dtype=np.int64),
    )

    lookup = _protocol_lookup(build_physical_query_masks(k_path))

    assert lookup["unseen_structure_physical_k_interpolation"].name == "unseen_structure_physical_k_interpolation"
    assert lookup["unseen_structure_physical_heldout_segment"].name == "unseen_structure_physical_heldout_segment"


def test_query_benchmark_restores_complete_case_and_resumes_epoch_checkpoint(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    dataset = tmp_path / "query_tiny.npz"
    config = tmp_path / "query.yaml"
    runs_dir = tmp_path / "runs"
    _write_tiny_query_npz(dataset)
    _write_query_config(config, dataset, runs_dir)

    run_dir = run_query_benchmark(config)
    rows = _history_rows(run_dir)
    assert len(rows) == 1
    assert rows[0]["resumed_from_case_artifact"] == "False"
    checkpoint = run_dir / "cases" / "unseen_structure_all_k_mlp_query_seed0.pt"
    assert checkpoint.is_file()

    resume_config = tmp_path / "query_resume.yaml"
    _write_query_config(resume_config, dataset, runs_dir, run_dir=run_dir)
    restored_run = run_query_benchmark(resume_config)
    restored_rows = _history_rows(restored_run)
    assert restored_run == run_dir
    assert restored_rows[0]["resumed_from_case_artifact"] == "True"
    assert json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["case_rows"] == 1

    checkpoint.unlink()
    epoch_resume_config = tmp_path / "query_epoch_resume.yaml"
    _write_query_config(epoch_resume_config, dataset, runs_dir, run_dir=run_dir, epochs=2)
    epoch_resumed_run = run_query_benchmark(epoch_resume_config)
    epoch_rows = _history_rows(epoch_resumed_run)

    assert epoch_rows[0]["resumed_from_epoch_checkpoint"] == "True"
    assert epoch_rows[0]["resumed_checkpoint_epoch"] == "1"
    assert checkpoint.is_file()


def test_query_fair_selection_keeps_learning_rates_separate_and_tests_three_seeds(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    dataset = tmp_path / "query_tiny.npz"
    config = tmp_path / "query_fair.yaml"
    _write_tiny_query_npz(dataset)
    config.write_text(
        "\n".join(
            [
                "experiment:", "  experiment_id: unit_query_fair", "  stage: query_a1_fair_selection",
                "dataset:", f"  path: {dataset.as_posix()}",
                "fair_selection:", "  enabled: true", "  expected_split_sizes: {train: 4, val: 1, test: 1}",
                "query:", "  requires_physical_k_coordinates: false", "  allow_index_query: true",
                "protocols: [unseen_structure_all_k]",
                "model:", "  candidates: [mlp_query]", "  params_by_model:", "    mlp_query: {hidden_dim: 8, depth: 1}",
                "training:", "  epochs: 1", "  checkpoint_epochs: [1]", "  learning_rates: [0.001, 0.0003]", "  seeds: [0, 1, 2]", "  batch_size: 4", "  device: cpu",
                "runtime:", "  device: cpu", "  require_accelerator: false",
                "output:", f"  runs_dir: {(tmp_path / 'runs').as_posix()}", "  write_query_report: false", "  write_dataset_audit_report: false", "",
            ]
        ), encoding="utf-8",
    )
    run_dir = run_query_benchmark(config)
    rows = _history_rows(run_dir)
    leaderboard = list(csv.DictReader((run_dir / "leaderboard.csv").open(encoding="utf-8")))
    manifest = json.loads((run_dir / "validation_selection_manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 3
    assert {row["evaluation_split"] for row in rows} == {"test"}
    assert leaderboard[0]["n_seeds"] == "3"
    assert manifest["test_driven_selection"] is False
