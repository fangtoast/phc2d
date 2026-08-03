"""Tests for shared run helpers."""

from __future__ import annotations

from datetime import datetime

from phc2d.runs import (
    RUN_META_NAME,
    add_throughput_metrics,
    make_run_dir,
    measure_wall_time,
    write_results,
    write_run_meta,
)


def test_run_dir_and_standard_files(tmp_path):
    run_dir = make_run_dir(
        "unit_task",
        runs_dir=tmp_path,
        timestamp=datetime(2026, 6, 11, 8, 30),
    )

    assert run_dir.name == "unit_task_202606110830"
    assert run_dir.parent.name == "20260611"
    write_run_meta(run_dir, {"script": "test"})
    write_results(run_dir, {"ok": True})
    assert (run_dir / RUN_META_NAME).is_file()
    assert (run_dir / "results.json").is_file()


def test_runtime_metric_helpers():
    metrics = {}
    with measure_wall_time(metrics, "wall_time_total_s"):
        sum(range(10))
    assert metrics["wall_time_total_s"] >= 0.0

    add_throughput_metrics(metrics, count=4, elapsed_s=2.0, prefix="samples")
    assert metrics["samples_per_s"] == 2.0
    assert metrics["time_per_sample_ms"] == 500.0
