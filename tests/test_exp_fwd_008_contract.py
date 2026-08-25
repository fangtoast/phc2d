"""Contract tests for EXP-FWD-008 fixed-200 training and band metrics.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from phc2d.forward import full_cell_six_domain_n40 as six
from phc2d.forward import full_cell_training_campaign as training
from phc2d.forward.full_cell import _convergence_record
from phc2d.forward import full_cell_dual_lane as dual
from phc2d.forward.full_cell_dual_lane import _dual_lane_convergence_record
from phc2d.forward.full_cell_metrics import compute_band_metrics, paired_bootstrap_mre


ROOT = Path(__file__).resolve().parents[1]


def _audit(lane: str) -> dict[str, object]:
    return {"lane": lane, "dataset_sha256": lane * 64, "idx_train_hash": "t" * 64, "idx_val_hash": "v" * 64, "idx_test_hash": "e" * 64, "k_points_sha256": "k" * 64}


def test_exp_fwd_008_configs_freeze_fixed_200_contract() -> None:
    profile = yaml.safe_load((ROOT / "configs/research/exp_fwd_008_k5_campaign_profile_v1.yaml").read_text(encoding="utf-8"))
    assert profile["experiment_id"] == "EXP-FWD-008"
    assert profile["training"] == {
        **profile["training"],
        "seed": 0,
    }
    assert profile["training"]["required_stop_epoch"] == 200
    assert profile["training"]["max_epochs"] == 200
    assert profile["training"]["early_stopping_patience"] == 0
    assert profile["training"]["epoch_100_continue_gate"] is False
    assert profile["training"]["convergence_window"] == [181, 200]
    assert profile["training"]["convergence_relative_threshold"] == 0.005
    assert profile["training"]["require_terminal_checkpoint"] is True
    assert profile["datasets"]["B"]["sha256"] == "7d2c6bef128c7ee61e0c5ca0eb9561b95992ccf8e53f325274c4889752ea86b6"
    assert profile["datasets"]["G"]["sha256"] == "eb140534d941cbdb53addca279f00337810124a34fade20a19bfb969e0e6af20"
    for name in ("binary", "graded", "dual_lane"):
        config = yaml.safe_load((ROOT / f"configs/research/full_cell_k5_forward_008_{name}_seed0_v1.yaml").read_text(encoding="utf-8"))
        assert config["experiment"]["experiment_id"] == "EXP-FWD-008"
        assert config["training"]["max_epochs"] == 200
        assert config["training"]["early_stopping_patience"] == 0
        assert config["training"]["epoch_100_continue_gate"] is False
        assert config["training"]["require_terminal_checkpoint"] is True


def test_exp_fwd_008_keeps_exact_17_tasks_and_three_waves(tmp_path: Path) -> None:
    profile = yaml.safe_load((ROOT / "configs/research/exp_fwd_008_k5_campaign_profile_v1.yaml").read_text(encoding="utf-8"))
    rows = training._task_rows(tmp_path, profile, {"B": _audit("B"), "G": _audit("G")}, stage="FORMAL")
    assert len(rows) == 17
    assert [row["wave"] for row in rows].count("W1") == 8
    assert [row["wave"] for row in rows].count("W2") == 8
    assert [row["wave"] for row in rows].count("W3") == 1
    assert rows[-1]["train_lane"] == "J"
    assert training._admission_submission_units(profile, rows) == (15, 2)


def test_fixed_convergence_formula_and_zero_denominator() -> None:
    history = [{"epoch": epoch, "val_bands_2_10_mre_pct": 0.0} for epoch in range(1, 201)]
    result = _convergence_record(history)
    assert result["relative_change"] == 0.0
    assert result["passed"] is True
    history[-1]["val_bands_2_10_mre_pct"] = 1.0
    result = _convergence_record(history)
    assert result["relative_change"] > 0.005
    with pytest.raises(ValueError, match="missing epochs"):
        _convergence_record(history[:-1])
    history[180]["val_bands_2_10_mre_pct"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _convergence_record(history)


def test_dual_lane_convergence_record_is_json_serializable() -> None:
    history = [
        {
            "epoch": epoch,
            "binary_val_bands_2_10_mre_pct": 1.0,
            "graded_val_bands_2_10_mre_pct": 2.0,
            "val_bands_2_10_mre_pct": 1.5,
        }
        for epoch in range(1, 201)
    ]
    result = _dual_lane_convergence_record(
        history,
        threshold=0.005,
        contract_metric="validation_bands_2_10_mre",
    )

    import json

    json.dumps(result)
    assert set(result["curves"]) == {"binary", "graded", "equal_lane"}
    assert result["curves"]["equal_lane"] is not result
    assert all(curve["passed"] for curve in result["curves"].values())


def test_dual_lane_fixed_200_finalizer_recovery_preserves_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import csv
    import json
    import torch

    run = tmp_path / "joint"
    (run / "figures").mkdir(parents=True)
    (run / "run.log").write_text("original training log\n", encoding="utf-8")
    (run / "run.meta.json").write_text("{}\n", encoding="utf-8")
    config = {
        "datasets": {"binary": {}, "graded": {}},
        "model": {"name": "dual_lane_band_operator_v1", "params": {"head_kind": "dense"}},
        "training": {
            "required_stop_epoch": 200,
            "convergence_relative_threshold": 0.005,
            "convergence_metric": "validation_bands_2_10_mre",
            "seed": 0,
        },
    }
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    fields = [
        "epoch",
        "val_bands_2_10_mre_pct",
        "binary_val_bands_2_10_mre_pct",
        "graded_val_bands_2_10_mre_pct",
        "wall_time_s",
        "optimizer_steps",
        "samples_seen",
    ]
    with (run / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, 201):
            writer.writerow(
                {
                    "epoch": epoch,
                    "val_bands_2_10_mre_pct": 1.5,
                    "binary_val_bands_2_10_mre_pct": 1.0,
                    "graded_val_bands_2_10_mre_pct": 2.0,
                    "wall_time_s": epoch,
                    "optimizer_steps": epoch * 10,
                    "samples_seen": epoch * 64,
                }
            )
    identity = {
        "canonical_model_name": "dual_lane_band_operator_v1",
        "factory_function_identity": "factory",
        "state_dict_schema_hash": "schema",
        "architecture_fingerprint": "fingerprint",
        "parameter_count": 3,
    }
    torch.save(
        {
            "epoch": 7,
            "model_name": "dual_lane_band_operator_v1",
            "model_identity": identity,
            "input_shape": (64, 64),
            "output_shape": (81, 10),
        },
        run / "model.pt",
    )
    torch.save({"epoch": 200}, run / "epoch200.pt")
    source_hashes = {
        name: dual._sha256_checkpoint(run / name) for name in ("model.pt", "epoch200.pt")
    }
    monkeypatch.setattr(
        dual,
        "_lane_contracts",
        lambda resolved: {
            "binary": {"selected_train": range(8), "idx_val": range(2)},
            "graded": {"selected_train": range(9), "idx_val": range(3)},
        },
    )

    dual.recover_fixed_200_finalizer(run, source_slurm_job_id="123")

    assert source_hashes == {
        name: dual._sha256_checkpoint(run / name) for name in ("model.pt", "epoch200.pt")
    }
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    recovery = json.loads((run / "finalizer_recovery.json").read_text(encoding="utf-8"))
    assert metrics["binary_n_train"] == 8
    assert metrics["graded_n_val"] == 3
    assert results["status"] == "dual_lane_training_success"
    assert recovery["source_slurm_job_id"] == "123"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        dual.recover_fixed_200_finalizer(run)


def test_band_slices_metrics_zero_variance_and_reproducible_bootstrap() -> None:
    target = np.ones((8, 3, 10), dtype=np.float32)
    prediction = target.copy()
    prediction[:, :, 0] += 0.1
    prediction[:, :, 2] += 0.2
    metrics, samples, per_band, per_k = compute_band_metrics(prediction, target, bootstrap_replicates=128, bootstrap_seed=7)
    assert set(("band1", "bands2_10", "bands1_10")) == set(samples)
    assert metrics["band1_mre"] == pytest.approx(10.0)
    assert metrics["band1_r2_variance_weighted"] is None
    assert metrics["band1_r2_status"] == "undefined_zero_target_variance"
    assert metrics["band1_band_order_violation_status"] == "not_applicable_single_band"
    assert len(per_band) == 10
    assert len(per_k) == 3 * 3
    first = paired_bootstrap_mre(samples, samples, seed=11, replicates=128)
    second = paired_bootstrap_mre(samples, samples, seed=11, replicates=128)
    assert first == second
    assert first["bands2_10_mre_delta"] == pytest.approx(0.0)


def test_exp_fwd_008_counts_and_full_domain_labels() -> None:
    assert six.EXPECTED_CHECKPOINTS == 85
    assert six.EXPECTED_CELLS == 510
    assert six.EXPECTED_PREFLIGHTS == 54
    assert six.EXPECTED_SHARDS == 10
    config = yaml.safe_load((ROOT / "configs/research/full_cell_k5_six_domain_exp_fwd_008_n40_v1.yaml").read_text(encoding="utf-8"))
    assert [config["datasets"][domain]["label"] for domain in six.DOMAINS] == [
        "K5 Binary 测试集", "K5 Graded 测试集", "K20 Binary 测试集", "K20 Graded 测试集", "F50 Binary 测试集", "F50 Graded 测试集"
    ]


def test_legacy_inventory_relative_checkpoint_uses_inventory_repository_root(tmp_path: Path) -> None:
    current_root = tmp_path / "new-root"
    legacy_root = tmp_path / "legacy-root"
    inventory = legacy_root / "runs" / "20260810" / "campaign" / "checkpoint_inventory.csv"
    checkpoint = legacy_root / "runs" / "20260807" / "model" / "model.pt"
    inventory.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    current_root.mkdir()
    resolved = six._checkpoint_path(
        current_root,
        {"checkpoint_path": "runs/20260807/model/model.pt"},
        inventory_root=six._inventory_repository_root(inventory),
    )
    assert resolved == checkpoint


def test_exp_fwd_008_job_contract_is_read_only_on_failure_and_30_minute_watch() -> None:
    root = ROOT
    submit = (root / "jobs/paracloud_submit_exp_fwd_008_k5.sh").read_text(encoding="utf-8")
    status = (root / "jobs/paracloud_exp_fwd_008_status.sh").read_text(encoding="utf-8")
    assert "PHC2D_AUTHORIZE_EXP_FWD_008" in submit
    assert "submission_failure.txt" in submit
    assert "no automatic retry" in submit
    assert "sleep 1800" in status
    assert "auto_retry: false" in (root / "configs/research/exp_fwd_008_k5_campaign_profile_v1.yaml").read_text(encoding="utf-8")
