"""Tests for native phc2d forward train/eval smoke execution.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import importlib.util
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from phc2d.artifacts import SURROGATE_PRED_BANDS_NAME, TARGET_BANDS_NAME
from phc2d.forward import (
    load_forward_run_config,
    run_forward_benchmark,
    run_forward_eval,
    run_forward_train,
)
from phc2d.forward.runner import _write_benchmark_report
from phc2d.forward.runner import _run_forward_train
from phc2d.models import ModelSpec, get_model_spec, register_model


def _write_tiny_npz(path: Path) -> None:
    rng = np.random.default_rng(123)
    rho = rng.random((8, 10), dtype=np.float32)
    weights = rng.normal(size=(10, 6), scale=0.05).astype(np.float32)
    wh = (rho @ weights).reshape(8, 3, 2).astype(np.float32)
    np.savez(
        path,
        rho=rho,
        Wh=wh,
        idx_train=np.arange(0, 5, dtype=np.int64),
        idx_val=np.arange(5, 6, dtype=np.int64),
        idx_test=np.arange(6, 8, dtype=np.int64),
        config_yaml="tiny: true\n",
    )


def _write_config(path: Path, dataset_path: Path, runs_dir: Path, *, checkpoint: Path | None = None) -> None:
    checkpoint_line = f"  checkpoint_path: {checkpoint.as_posix()}\n" if checkpoint else ""
    path.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_forward_smoke",
                "dataset:",
                f"  path: {dataset_path.as_posix()}",
                "  input_repr: rho_wedge",
                "  output_target: Wh",
                "model:",
                "  name: res_mlp",
                "  params:",
                "    hidden_dim: 8",
                "    depth: 1",
                checkpoint_line.rstrip(),
                "training:",
                "  epochs: 1",
                "  checkpoint_epochs: [1]",
                "  batch_size: 2",
                "  learning_rate: 0.001",
                "  seed: 3",
                "  device: cpu",
                "evaluation:",
                "  split: test",
                "metrics:",
                "  relative_error_floor: 1.0e-8",
                "output:",
                f"  runs_dir: {runs_dir.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_near_field_config(path: Path, dataset_path: Path, runs_dir: Path, report_path: Path | None = None) -> None:
    report_path = report_path or (runs_dir / "near_field_unit_feasibility.md")
    path.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_near_field_smoke",
                "  execution: schema_only",
                "dataset:",
                f"  path: {dataset_path.as_posix()}",
                "  input_repr: field_conditioned",
                "  output_target: near_field_te_tm",
                "model:",
                "  name: unet_field",
                "training:",
                "  enabled: false",
                "  execution: schema_only",
                "  epochs: 0",
                "  batch_size: 1",
                "  device: cpu",
                "evaluation:",
                "  split: test",
                "metrics:",
                "  relative_error_floor: 1.0e-8",
                "output:",
                f"  runs_dir: {runs_dir.as_posix()}",
                f"  report: {report_path.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _assert_standard_artifacts(run_dir: Path) -> None:
    for name in [
        "run.log",
        "run.meta.json",
        "config.resolved.yaml",
        "results.json",
        "metrics.json",
        "history.csv",
    ]:
        assert (run_dir / name).is_file(), name
    assert (run_dir / "figures").is_dir()


def test_representation_ablation_config_parses_to_trainable_default() -> None:
    config = load_forward_run_config("configs/research/representation_ablation.yaml")
    assert config.dataset_path == Path("data/paper_like_64_mpb.npz")
    assert config.representation.input_repr == "rho_wedge"
    assert config.representation.output_target == "Wh"
    assert config.model_name == "res_mlp"


def _assert_benchmark_artifacts(run_dir: Path) -> None:
    for name in [
        "run.log",
        "run.meta.json",
        "config.resolved.yaml",
        "results.json",
        "metrics.json",
        "history.csv",
        "leaderboard.csv",
        "per_band_metrics.csv",
        "integrity_summary.json",
        "model_identity_summary.json",
        "figures",
    ]:
        if name in {"figures"}:
            assert (run_dir / name).is_dir(), name
        else:
            assert (run_dir / name).is_file(), name


def test_forward_train_creates_standard_run_artifacts(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    dataset = tmp_path / "tiny.npz"
    config_path = tmp_path / "train.yaml"
    runs_dir = tmp_path / "runs"
    _write_tiny_npz(dataset)
    _write_config(config_path, dataset, runs_dir)

    run_dir = run_forward_train(config_path)

    assert run_dir.parent.name == "20260617" or run_dir.parent.name.isdigit()
    _assert_standard_artifacts(run_dir)
    assert (run_dir / "model.pt").is_file()
    assert (run_dir / "checkpoint_epoch_1.pt").is_file()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"mse", "mae", "rmse", "wall_time_total_s", "train_time_s"}.issubset(metrics)
    assert metrics["device_name"] == "cpu"
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["smoke_only_no_mpb_meep"] is True
    target_bands = np.load(run_dir / TARGET_BANDS_NAME)
    pred_bands = np.load(run_dir / SURROGATE_PRED_BANDS_NAME)
    assert target_bands.shape == pred_bands.shape == (2, 3, 2)
    assert results["band_array_artifacts"]["mpb_replay_bands"] is None


def test_forward_train_resumes_from_epoch_checkpoint(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    dataset = tmp_path / "tiny.npz"
    config_path = tmp_path / "train.yaml"
    runs_dir = tmp_path / "runs"
    _write_tiny_npz(dataset)
    _write_config(config_path, dataset, runs_dir)

    initial_run = run_forward_train(config_path)
    checkpoint = initial_run / "checkpoint_epoch_1.pt"
    assert checkpoint.is_file()
    for name in ("model.pt", "metrics.json", "results.json", "history.csv"):
        path = initial_run / name
        if path.exists():
            path.unlink()

    resumed_config_path = tmp_path / "train_resume.yaml"
    _write_config(resumed_config_path, dataset, runs_dir)
    payload = resumed_config_path.read_text(encoding="utf-8").replace("  epochs: 1", "  epochs: 2")
    resumed_config_path.write_text(payload, encoding="utf-8")
    config = load_forward_run_config(resumed_config_path)

    resumed_run = _run_forward_train(config, run_dir_override=initial_run)

    assert resumed_run == initial_run
    metrics = json.loads((resumed_run / "metrics.json").read_text(encoding="utf-8"))
    results = json.loads((resumed_run / "results.json").read_text(encoding="utf-8"))
    history = (resumed_run / "history.csv").read_text(encoding="utf-8")
    assert metrics["resumed_from_checkpoint"] is True
    assert metrics["resumed_checkpoint_epoch"] == 1
    assert results["resumed_from_checkpoint"] is True
    assert "2," in history


def test_forward_config_parses_formal_training_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.npz"
    config_path = tmp_path / "formal.yaml"
    _write_tiny_npz(dataset)
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_forward_formal_fields",
                "dataset:",
                f"  path: {dataset.as_posix()}",
                "  input_repr: rho_wedge",
                "  output_target: Wh",
                "model:",
                "  name: res_mlp_wide",
                "training:",
                "  max_epochs: 3",
                "  checkpoint_epochs: [1, 3]",
                "  learning_rates: [0.001, 0.0003, 0.0001]",
                "  optimizer: adamw",
                "  weight_decay: 1.0e-5",
                "  loss:",
                "    name: band_aware",
                "    relative_weight: 0.25",
                "output:",
                f"  runs_dir: {(tmp_path / 'runs').as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_forward_run_config(config_path)

    assert config.epochs == 3
    assert config.checkpoint_epochs == (1, 3)
    assert config.learning_rates == (0.001, 0.0003, 0.0001)
    assert config.optimizer == "adamw"
    assert config.weight_decay == pytest.approx(1.0e-5)
    assert config.loss_name == "band_aware"
    assert config.loss_params["relative_weight"] == pytest.approx(0.25)


def test_forward_eval_loads_train_checkpoint(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    dataset = tmp_path / "tiny.npz"
    train_config = tmp_path / "train.yaml"
    eval_config = tmp_path / "eval.yaml"
    runs_dir = tmp_path / "runs"
    _write_tiny_npz(dataset)
    _write_config(train_config, dataset, runs_dir)
    train_run = run_forward_train(train_config)

    checkpoint = train_run / "model.pt"
    _write_config(eval_config, dataset, runs_dir, checkpoint=checkpoint)
    eval_run = run_forward_eval(eval_config)

    _assert_standard_artifacts(eval_run)
    metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    assert {"mse", "mae", "rmse", "eval_time_s", "inference_time_per_sample_ms"}.issubset(metrics)
    assert metrics["checkpoint_loaded"] is True
    target_bands = np.load(eval_run / TARGET_BANDS_NAME)
    pred_bands = np.load(eval_run / SURROGATE_PRED_BANDS_NAME)
    assert target_bands.shape == pred_bands.shape == (2, 3, 2)


def test_forward_eval_discovers_latest_checkpoint_for_same_config(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    dataset = tmp_path / "tiny.npz"
    config_path = tmp_path / "train_eval.yaml"
    runs_dir = tmp_path / "runs"
    _write_tiny_npz(dataset)
    _write_config(config_path, dataset, runs_dir)

    train_run = run_forward_train(config_path)
    eval_run = run_forward_eval(config_path)

    metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    resolved = (eval_run / "config.resolved.yaml").read_text(encoding="utf-8")
    assert metrics["checkpoint_loaded"] is True
    assert str(train_run / "model.pt") in resolved


def test_forward_benchmark_runs_matrix_and_writes_reports(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    # use a small local clone of the benchmark profile so we don't mutate global runs
    local_config = tmp_path / "representation_ablation.yaml"
    tmp_profile = (
        "experiment:\n"
        "  experiment_id: representation_ablation_benchmark\n"
        "  family: research\n"
        "  stage: forward_benchmark\n"
        "  profile_only: true\n"
        "dataset:\n"
        "  path: data/paper_like_64_mpb.npz\n"
        "  input_repr: rho_wedge\n"
        "  output_target: Wh\n"
        "benchmark:\n"
        "  - input_repr: rho_wedge\n"
        "    model: res_mlp\n"
        "  - input_repr: rho_grid\n"
        "    model: lifted_cnn\n"
        "  - input_repr: epsilon_grid\n"
        "    model: lifted_cnn\n"
        "  - input_repr: epsilon_grid\n"
        "    model: fno_lite\n"
        "model:\n"
        "  candidates:\n"
        "    - res_mlp\n"
        "    - lifted_cnn\n"
        "    - fno_lite\n"
        "  overrides:\n"
        "    hidden_dim: 16\n"
        "    depth: 2\n"
        "    width: 12\n"
        "    channels: 8\n"
        "    modes: [6, 6]\n"
        "training:\n"
        "  epochs: 1\n"
        "  batch_size: 2\n"
        "  device: cpu\n"
        "metrics:\n"
        "  relative_error_floor: 1.0e-8\n"
        "output:\n"
        f"  runs_dir: {tmp_path.as_posix()}/runs\n"
    )
    local_config.write_text(tmp_profile, encoding="utf-8")

    run_dir = run_forward_benchmark(local_config)
    _assert_benchmark_artifacts(run_dir)
    report_path = Path("docs/results") / f"representation_benchmark_{datetime.now().strftime('%Y%m%d')}.md"
    assert report_path.is_file()
    assert (run_dir / "leaderboard.csv").stat().st_size > 0
    identity_summary = json.loads((run_dir / "model_identity_summary.json").read_text(encoding="utf-8"))
    assert len(identity_summary["identity_rows"]) == 4
    assert identity_summary["unique_architecture_fingerprints"]
    leaderboard = (run_dir / "leaderboard.csv").read_text(encoding="utf-8")
    assert "canonical_model_name" in leaderboard
    assert "architecture_fingerprint" in leaderboard
    first_case = next((run_dir / "cases").glob("*/eval"))
    case_meta = json.loads((first_case / "run.meta.json").read_text(encoding="utf-8"))
    case_results = json.loads((first_case / "results.json").read_text(encoding="utf-8"))
    assert "model_identity" in case_meta
    assert "model_identity" in case_results


def test_forward_benchmark_report_honors_configured_path(tmp_path: Path) -> None:
    source = tmp_path / "tiny.npz"
    _write_tiny_npz(source)
    report_path = tmp_path / "reports" / "custom_forward_report.md"
    config_path = tmp_path / "forward_report.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: unit_forward_report",
                "dataset:",
                f"  path: {source.as_posix()}",
                "  input_repr: rho_wedge",
                "  output_target: Wh",
                "model:",
                "  name: res_mlp",
                "training:",
                "  epochs: 1",
                "  batch_size: 2",
                "  device: cpu",
                "output:",
                f"  runs_dir: {(tmp_path / 'runs').as_posix()}",
                f"  report: {report_path.as_posix()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config = load_forward_run_config(config_path)

    written = _write_benchmark_report(
        config,
        [
            {
                "case_id": "rho_wedge_res_mlp_1",
                "input_repr": "rho_wedge",
                "model": "res_mlp",
                "mse": 0.1,
                "mae": 0.2,
                "rmse": 0.3,
                "avg_MRE_bands_2_10_pct": 4.0,
            }
        ],
        [],
        completed_at=datetime(2026, 6, 21),
    )

    assert written == report_path
    assert report_path.is_file()


def test_forward_fair_selection_uses_validation_and_three_seed_test_summary(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    dataset = tmp_path / "tiny.npz"
    config_path = tmp_path / "fair.yaml"
    _write_tiny_npz(dataset)
    config_path.write_text(
        "\n".join(
            [
                "experiment:", "  experiment_id: unit_forward_fair", "  stage: forward_a1_fair_selection",
                "dataset:", f"  path: {dataset.as_posix()}", "  input_repr: rho_wedge", "  output_target: Wh",
                "fair_selection:", "  enabled: true", "  expected_split_sizes: {train: 5, val: 1, test: 2}",
                "benchmark:", "  - input_repr: rho_wedge", "    model: res_mlp",
                "training:", "  epochs: 1", "  checkpoint_epochs: [1]", "  learning_rates: [0.001, 0.0003]", "  seeds: [0, 1, 2]", "  batch_size: 2", "  device: cpu",
                "runtime:", "  device: cpu", "  require_accelerator: false",
                "output:", f"  runs_dir: {(tmp_path / 'runs').as_posix()}", "",
            ]
        ), encoding="utf-8",
    )
    run_dir = run_forward_benchmark(config_path)
    summary = json.loads((run_dir / "integrity_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "validation_selection_manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((run_dir / "leaderboard.csv").open(encoding="utf-8")))
    assert summary["test_driven_selection"] is False
    assert summary["all_summary_rows_n3"] is True
    assert manifest["selections"][0]["seed_count"] == 3
    assert rows[0]["n_seeds"] == "3"


def test_forward_benchmark_rejects_alias_as_independent_case(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is optional and not installed")

    spec = get_model_spec("res_mlp")

    def _alias_factory(**kwargs):
        from phc2d.models import build_model

        return build_model("res_mlp", **kwargs)

    register_model(
        "unit_forward_alias_res_mlp",
        _alias_factory,
        replace=True,
        spec=ModelSpec(
            name="unit_forward_alias_res_mlp",
            family=spec.family,
            input_repr=spec.input_repr,
            output_target=spec.output_target,
            canonical_name="res_mlp",
            alias_of="res_mlp",
            requires=spec.requires,
            defaults=spec.defaults,
            source="unit test alias",
        ),
    )
    local_config = tmp_path / "alias_benchmark.yaml"
    local_config.write_text(
        "\n".join(
            [
                "experiment:",
                "  experiment_id: alias_benchmark",
                "dataset:",
                "  path: data/paper_like_64_mpb.npz",
                "  input_repr: rho_wedge",
                "  output_target: Wh",
                "benchmark:",
                "  - input_repr: rho_wedge",
                "    model: unit_forward_alias_res_mlp",
                "training:",
                "  epochs: 1",
                "  batch_size: 2",
                "  device: cpu",
                "output:",
                f"  runs_dir: {tmp_path.as_posix()}/runs",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="alias models"):
        run_forward_benchmark(local_config)


def test_forward_train_schema_only_near_field_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("h5py") is None:
        pytest.skip("h5py is optional and not installed")

    source = Path("data/field_tiny_mpb.h5")
    if not source.exists():
        pytest.skip("data/field_tiny_mpb.h5 is not present in this checkout")

    config_path = tmp_path / "near_field_tiny_unet.yaml"
    runs_dir = tmp_path / "runs"
    custom_report = tmp_path / "near_field_unit_feasibility.md"
    _write_near_field_config(config_path, source, runs_dir, report_path=custom_report)

    run_dir = run_forward_train(config_path)

    assert run_dir.parent.name == "20260617" or run_dir.parent.name.isdigit()
    _assert_standard_artifacts(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "relative_l2_field_error" in metrics
    assert "field_inference_time_s" in metrics
    assert custom_report.is_file()
    assert (run_dir / "near_field_feasibility.md").is_file()
