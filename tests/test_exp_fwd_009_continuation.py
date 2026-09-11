"""Contracts for EXP-FWD-009 weight-only convergence continuation.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from phc2d.forward import full_cell, full_cell_continuation_campaign, full_cell_dual_lane
from phc2d.forward.full_cell_warm_start import (
    load_weight_only_warm_start,
    rolling_convergence_record,
    seed_derivative_best_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exp_fwd_009_configs_register_weight_only_continuation() -> None:
    for lane in ("binary", "graded"):
        path = ROOT / f"configs/research/full_cell_k5_forward_009_{lane}_seed0_v1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["experiment"]["experiment_id"] == "EXP-FWD-009"
        assert config["training"]["max_epochs"] == 600
        assert config["training"]["early_stopping_patience"] == 0
        assert "required_stop_epoch" not in config["training"]
        assert config["continuation"] == {
            **config["continuation"],
            "mode": "epoch200_weights_optimizer_reset",
            "source_epoch": 200,
            "optimizer_reset": True,
            "check_interval": 20,
            "relative_threshold": 0.005,
            "max_total_epoch": 600,
            "band1_blocking": False,
        }
        resolved = full_cell._validate_and_resolve(
            config,
            model_name="convnext_band_v1",
            seed=0,
            admission_probe=False,
        )
        assert resolved["training"]["learning_rate"] == pytest.approx(1.0e-3)
    joint_path = ROOT / "configs/research/full_cell_k5_forward_009_dual_lane_seed0_v1.yaml"
    joint = yaml.safe_load(joint_path.read_text(encoding="utf-8"))
    resolved_joint = full_cell_dual_lane._validate_and_resolve(
        joint,
        seed=0,
        admission_probe=False,
        head_kind="dense",
    )
    assert resolved_joint["training"]["learning_rate"] == pytest.approx(5.0e-4)
    assert resolved_joint["continuation"]["joint_required_curves"] == [
        "binary",
        "graded",
        "equal_lane",
    ]


def test_weight_only_source_and_rolling_convergence_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    fields = ["epoch", "val_bands_2_10_mre_pct", "optimizer_steps", "samples_seen", "wall_time_s"]
    with (source / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 201):
            writer.writerow(
                {
                    "epoch": epoch,
                    "val_bands_2_10_mre_pct": 1.0,
                    "optimizer_steps": epoch * 2,
                    "samples_seen": epoch * 64,
                    "wall_time_s": epoch,
                }
            )
    (source / "config.resolved.yaml").write_text("experiment: {experiment_id: EXP-FWD-008}\n", encoding="utf-8")
    torch.save({"epoch": 7}, source / "model.pt")
    torch.save({"epoch": 200, "state_dict": {"weight": torch.ones(1)}}, source / "epoch200.pt")
    result = {
        "model": "convnext_band_v1",
        "seed": 0,
        "stop_epoch": 200,
        "history_epoch_count": 200,
        "best_epoch": 7,
        "checkpoint_sha256": _sha256(source / "model.pt"),
        "terminal_checkpoint_sha256": _sha256(source / "epoch200.pt"),
    }
    (source / "results.json").write_text(json.dumps(result), encoding="utf-8")

    loaded = load_weight_only_warm_start(
        source,
        expected_model="convnext_band_v1",
        expected_seed=0,
        torch_module=torch,
    )
    assert loaded["optimizer_state_restored"] is False
    assert loaded["data_order_state_restored"] is False
    derivative = tmp_path / "derivative"
    derivative.mkdir()
    copied = seed_derivative_best_checkpoint(loaded, derivative)
    assert _sha256(copied) == result["checkpoint_sha256"]
    convergence = rolling_convergence_record(
        loaded["source_history"],
        end_epoch=200,
        metric="val_bands_2_10_mre_pct",
        threshold=0.005,
    )
    assert convergence["first_window"] == [181, 190]
    assert convergence["second_window"] == [191, 200]
    assert convergence["passed"] is True


def test_weight_only_warm_start_accepts_only_full_cell_resolved_source_id(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    fields = ["epoch", "val_bands_2_10_mre_pct", "optimizer_steps", "samples_seen", "wall_time_s"]
    with (source / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 201):
            writer.writerow(
                {
                    "epoch": epoch,
                    "val_bands_2_10_mre_pct": 1.0,
                    "optimizer_steps": epoch,
                    "samples_seen": epoch,
                    "wall_time_s": epoch,
                }
            )
    torch.save({"epoch": 7}, source / "model.pt")
    torch.save({"epoch": 200, "state_dict": {"weight": torch.ones(1)}}, source / "epoch200.pt")
    result = {
        "model": "convnext_band_v1",
        "seed": 0,
        "stop_epoch": 200,
        "history_epoch_count": 200,
        "checkpoint_sha256": _sha256(source / "model.pt"),
        "terminal_checkpoint_sha256": _sha256(source / "epoch200.pt"),
    }
    (source / "results.json").write_text(json.dumps(result), encoding="utf-8")

    (source / "config.resolved.yaml").write_text(
        "experiment: {experiment_id: EXP-FWD-020_n40000_e200}\n", encoding="utf-8"
    )
    loaded = load_weight_only_warm_start(
        source,
        expected_model="convnext_band_v1",
        expected_seed=0,
        torch_module=torch,
        expected_source_experiment="EXP-FWD-020",
    )
    assert loaded["source_experiment_id"] == "EXP-FWD-020_n40000_e200"

    (source / "config.resolved.yaml").write_text(
        "experiment: {experiment_id: EXP-FWD-020-other_n40000_e200}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="experiment identity mismatch"):
        load_weight_only_warm_start(
            source,
            expected_model="convnext_band_v1",
            expected_seed=0,
            torch_module=torch,
            expected_source_experiment="EXP-FWD-020",
        )


def test_exp_fwd_009_freezes_exact_five_task_derivative(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for model, _lane, source_task, _config in full_cell_continuation_campaign.TASKS:
        run = source / source_task
        run.mkdir()
        (run / "history.csv").write_text("epoch\n200\n", encoding="utf-8")
        (run / "model.pt").write_bytes(b"best")
        (run / "epoch200.pt").write_bytes(b"terminal")
        (run / "results.json").write_text(
            json.dumps({"model": model, "stop_epoch": 200}), encoding="utf-8"
        )
    campaign = tmp_path / "campaign"
    full_cell_continuation_campaign.freeze_campaign(
        source,
        campaign,
        repository_root=ROOT,
    )
    with (campaign / "task_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert [row["lane"] for row in rows] == ["B", "G", "B", "G", "J"]
    assert len({row["source_run"] for row in rows}) == 5
    freeze = json.loads((campaign / "freeze.json").read_text(encoding="utf-8"))
    assert freeze["continuation_mode"] == "epoch200_weights_optimizer_reset"
    assert freeze["max_total_epoch"] == 600


def test_exp_fwd_009_freezes_12_plus_5_evaluation_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    continuation = tmp_path / "continuation"
    source.mkdir()
    continuation.mkdir()
    models = [
        "mlp", "res_mlp_wide", "lifted_cnn", "fno2d_band_v2",
        "mlp_mixer_band_v1", "res_mlp_gated_v2", "convnext_band_v1",
        "tfno2d_band_v3",
    ]
    identity = {
        "canonical_model_name": "model",
        "factory_function_identity": "factory",
        "state_dict_schema_hash": "schema",
        "architecture_fingerprint": "architecture",
        "parameter_count": 1,
    }
    base_rows = []
    for index, (model, lane) in enumerate(
        [(model, lane) for model in models for lane in ("B", "G")]
        + [("dual_lane_band_operator_v1", "J")]
    ):
        run = source / f"formal_{index:02d}_{model}_{lane}_s0"
        run.mkdir()
        (run / "model.pt").write_bytes(f"base-{index}".encode())
        source_status = "formal_training_success" if index == 2 else "training_complete_convergence_not_met"
        (run / "results.json").write_text(
            json.dumps({"status": source_status, "stop_epoch": 200, "model_identity": identity}),
            encoding="utf-8",
        )
        base_rows.append({"task_index": index, "task_id": run.name, "model": model, "train_lane": lane, "seed": 0, "run_dir": str(run)})
    with (source / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_rows[0])); writer.writeheader(); writer.writerows(base_rows)

    continuation_rows = []
    for index, (model, lane, _source_task, _config) in enumerate(full_cell_continuation_campaign.TASKS):
        run = continuation / f"continue_{index:02d}_{model}_{lane}_s0"
        run.mkdir()
        (run / "model.pt").write_bytes(f"continued-{index}".encode())
        (run / "results.json").write_text(
            json.dumps({"status": "continuation_training_success", "stopped_total_epoch": 220, "model_identity": {**identity, "canonical_model_name": model}}),
            encoding="utf-8",
        )
        continuation_rows.append({"task_index": index, "model": model, "lane": lane, "seed": 0, "run_dir": str(run)})
    with (continuation / "task_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(continuation_rows[0])); writer.writeheader(); writer.writerows(continuation_rows)

    output = full_cell_continuation_campaign.finalize_evaluation_inventory(source, continuation)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 17
    assert sum(row["evaluation_checkpoint_source_experiment"] == "EXP-FWD-009" for row in rows) == 5
    assert sum(row["evaluation_checkpoint_source_experiment"] == "EXP-FWD-008" for row in rows) == 12
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        full_cell_continuation_campaign.finalize_evaluation_inventory(source, continuation)


def test_exp_fwd_009_six_domain_contract_matches_007_matrix() -> None:
    config = yaml.safe_load((ROOT / "configs/research/full_cell_k5_six_domain_exp_fwd_009_n40_v1.yaml").read_text(encoding="utf-8"))
    assert config["experiment_id"] == "EXP-FWD-009"
    assert config["shards"] == {**config["shards"], "fixed_array": "0-9%8", "count": 10}
    assert config["statistics"]["paired_bootstrap_replicates"] == 10000
    submit = (ROOT / "jobs/paracloud_submit_exp_fwd_009_six_domain.sh").read_text(encoding="utf-8")
    assert "--array=0-9%8" in submit
    assert "afterok:$shard_id" in submit
    assert "PHC2D_AUTHORIZE_EXP_FWD_009" in submit
    assert "recover-finalizer" in submit
    assert "source_evaluation_cells_sha256" in submit
