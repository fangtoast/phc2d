"""Contracts and lightweight dataset adapters for dual-lane Full-Wh training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from phc2d.forward.full_cell import (
    _FullCellH5Dataset,
    _convergence_record,
    _read_contract,
)
from phc2d.forward.gap_aware_loss import resolve_gap_aware_loss_config
from phc2d.models import get_model_spec


LANES = ("binary", "graded")


def dual_lane_convergence_record(
    history: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    contract_metric: str,
) -> dict[str, Any]:
    """Build the three fixed-200 convergence records without shared containers."""

    curves = {
        "binary": _convergence_record(
            history,
            metric="binary_val_bands_2_10_mre_pct",
            first_window=(181, 190),
            second_window=(191, 200),
            threshold=threshold,
        ),
        "graded": _convergence_record(
            history,
            metric="graded_val_bands_2_10_mre_pct",
            first_window=(181, 190),
            second_window=(191, 200),
            threshold=threshold,
        ),
        "equal_lane": _convergence_record(
            history,
            metric="val_bands_2_10_mre_pct",
            first_window=(181, 190),
            second_window=(191, 200),
            threshold=threshold,
        ),
    }
    return {
        **curves["equal_lane"],
        "contract_metric": contract_metric,
        "curves": curves,
    }


def validate_and_resolve_dual_lane(
    config: Mapping[str, Any],
    *,
    seed: int,
    admission_probe: bool,
    head_kind: str,
) -> dict[str, Any]:
    """Resolve one dual-lane configuration without importing training stacks."""

    experiment = dict(config.get("experiment") or {})
    datasets = dict(config.get("datasets") or {})
    representation = dict(config.get("representation") or {})
    model_cfg = dict(config.get("model") or {})
    training = dict(config.get("training") or {})
    continuation = dict(config.get("continuation") or {})
    runtime = dict(config.get("runtime") or {})
    output = dict(config.get("output") or {})
    if set(datasets) != set(LANES):
        raise ValueError("datasets must contain exactly binary and graded")
    if int(seed) not in {0, 1, 2}:
        raise ValueError("dual-lane seeds are fixed to 0, 1, 2")
    if head_kind not in {"dense", "k_conditioned"}:
        raise ValueError("head_kind must be dense or k_conditioned")
    model_name = str(model_cfg.get("name", "dual_lane_band_operator_v1"))
    spec = get_model_spec(model_name)
    if not spec.accepts_lane_id or spec.input_layout != "image" or spec.output_target != "Wh":
        raise ValueError(f"model {model_name!r} is not a dual-lane Full-Wh image model")
    micro_batch = int(training.get("micro_batch_size", 64))
    if micro_batch <= 0 or micro_batch % 2:
        raise ValueError("dual-lane micro_batch_size must be a positive even number")
    if int(training.get("effective_batch_size", micro_batch)) != micro_batch:
        raise ValueError("dual-lane runner currently requires effective_batch_size=micro_batch_size")
    if str(training.get("optimizer", "")).lower() != "adamw":
        raise ValueError("dual-lane training requires AdamW")
    if str(runtime.get("precision", "")).lower() != "fp32":
        raise ValueError("dual-lane training requires FP32")
    if bool(runtime.get("amp_enabled")) or bool(runtime.get("tf32_enabled")):
        raise ValueError("dual-lane training forbids AMP and TF32")
    resolve_gap_aware_loss_config(training.get("loss"))
    max_epochs = int(training.get("max_epochs", 100))
    evidence_status = str(experiment.get("evidence_status", ""))
    short_screen = evidence_status == "single_seed_short_budget_screen"
    if short_screen:
        if admission_probe:
            raise ValueError("short-budget screening and admission_probe are mutually exclusive")
        if int(seed) != 0 or max_epochs != 20:
            raise ValueError("dual-lane short-budget screening requires seed=0 and max_epochs=20")
        if bool(training.get("epoch_100_continue_gate", False)):
            raise ValueError("dual-lane short-budget screening forbids the epoch-100 continuation gate")
        if int(training.get("early_stopping_patience", 0)) != 0:
            raise ValueError("dual-lane short-budget screening requires early_stopping_patience=0")
    elif not admission_probe and max_epochs not in ({100, 200, 300, 600} if continuation else {100, 200}):
        raise ValueError("formal dual-lane max_epochs is outside the registered contract")
    gate_enabled = bool(training.get("epoch_100_continue_gate", False))
    if not admission_probe and gate_enabled:
        if max_epochs != 200:
            raise ValueError("dual-lane epoch-100 continuation gate requires max_epochs=200")
        if float(training.get("continue_gate_min_relative_improvement", 0.005)) != 0.005:
            raise ValueError("preregistered continuation threshold must equal 0.005")
        if int(training.get("early_stopping_patience", 0)) != 15:
            raise ValueError("dual-lane epoch-100 continuation requires patience 15")
    if not admission_probe and training.get("required_stop_epoch") is not None:
        required_stop = int(training["required_stop_epoch"])
        if max_epochs != required_stop or required_stop != 200:
            raise ValueError("fixed-200 dual-lane training requires max_epochs=required_stop_epoch=200")
        if int(training.get("early_stopping_patience", -1)) != 0:
            raise ValueError("fixed-200 dual-lane training requires early_stopping_patience=0")
        if bool(training.get("epoch_100_continue_gate", False)):
            raise ValueError("fixed-200 dual-lane training forbids epoch_100_continue_gate")
        if not bool(training.get("require_terminal_checkpoint", False)):
            raise ValueError("fixed-200 dual-lane training requires a terminal checkpoint")
        if tuple(int(value) for value in training.get("convergence_window", [])) != (181, 200):
            raise ValueError("fixed-200 dual-lane training requires convergence_window=[181, 200]")
        if str(training.get("convergence_metric", "")) != "validation_bands_2_10_mre":
            raise ValueError("fixed-200 dual-lane training requires validation_bands_2_10_mre")
        if float(training.get("convergence_relative_threshold", -1.0)) != 0.005:
            raise ValueError("fixed-200 dual-lane convergence threshold must be 0.005")
    if not admission_probe and continuation:
        experiment_id = str(experiment.get("experiment_id"))
        if experiment_id not in {"EXP-FWD-009", "EXP-FWD-011"}:
            raise ValueError("weight-only dual-lane continuation requires a registered experiment")
        if str(continuation.get("mode")) != "epoch200_weights_optimizer_reset":
            raise ValueError("unsupported dual-lane continuation mode")
        if int(continuation.get("source_epoch", -1)) != 200:
            raise ValueError("dual-lane continuation source_epoch must equal 200")
        expected_check_interval = 20 if experiment_id == "EXP-FWD-009" else 100
        if int(continuation.get("check_interval", -1)) != expected_check_interval:
            raise ValueError(
                f"dual-lane continuation check_interval must equal {expected_check_interval}"
            )
        if float(continuation.get("relative_threshold", -1.0)) != 0.005:
            raise ValueError("dual-lane continuation relative_threshold must equal 0.005")
        if not bool(continuation.get("optimizer_reset", False)):
            raise ValueError("weight-only dual-lane continuation must declare optimizer_reset=true")
        expected_max_epochs = 600 if experiment_id == "EXP-FWD-009" else 300
        if max_epochs != expected_max_epochs or training.get("required_stop_epoch") is not None:
            raise ValueError(
                f"dual-lane continuation requires max_epochs={expected_max_epochs} and no required stop"
            )
        if int(continuation.get("max_total_epoch", -1)) != expected_max_epochs:
            raise ValueError("dual-lane continuation max_total_epoch mismatch")
        if experiment_id == "EXP-FWD-011":
            if str(continuation.get("source_experiment")) != "EXP-FWD-010":
                raise ValueError("EXP-FWD-011 must continue EXP-FWD-010")
            if bool(continuation.get("stop_when_converged", True)):
                raise ValueError("EXP-FWD-011 must run through total epoch 300")
        if int(training.get("early_stopping_patience", -1)) != 0:
            raise ValueError("dual-lane continuation requires early_stopping_patience=0")
    params = dict(model_cfg.get("params") or {})
    params["head_kind"] = head_kind
    experiment_id = str(experiment.get("experiment_id", "full_cell_forward_upgrade_dual_lane_10k_v1"))
    return {
        "experiment": {
            **experiment,
            "experiment_id": experiment_id,
            "lane": "joint",
            "seed": int(seed),
            "admission_probe": bool(admission_probe),
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
        },
        "datasets": {lane: dict(datasets[lane]) for lane in LANES},
        "representation": {**representation, "layout": "image"},
        "model": {"name": model_name, "params": params},
        "training": {
            **training,
            "seed": int(seed),
            "max_epochs": 1 if admission_probe else max_epochs,
            "max_train_batches": 1 if admission_probe else None,
        },
        "runtime": runtime,
        "output": output,
        "continuation": continuation,
    }


def read_dual_lane_contracts(resolved: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read and cross-check the two lane-specific HDF5 contracts."""

    contracts: dict[str, dict[str, Any]] = {}
    for lane in LANES:
        lane_resolved = {
            "dataset": resolved["datasets"][lane],
            "representation": resolved["representation"],
            "training": resolved["training"],
        }
        contracts[lane] = _read_contract(lane_resolved)
    reference = contracts["binary"]
    graded = contracts["graded"]
    for key in ("grid_shape", "target_shape", "k_points_sha256"):
        if reference[key] != graded[key]:
            raise ValueError(f"dual-lane {key} mismatch: {reference[key]!r} != {graded[key]!r}")
    if reference["k_points"] is None:
        raise ValueError("dual-lane datasets must contain k_points")
    return contracts


def make_dual_lane_dataset(
    contract: Mapping[str, Any],
    resolved: Mapping[str, Any],
    indices: Any,
) -> Any:
    """Create the existing lazy HDF5 dataset adapter for one lane."""

    return _FullCellH5Dataset(
        path=contract["path"],
        indices=indices,
        grid_field=contract["grid_field"],
        layout="image",
        sampling=str(resolved["representation"].get("sampling", "continuous")),
        threshold=resolved["representation"].get("threshold"),
        target_mean=contract["target_mean"],
        target_std=contract["target_std"],
    )


__all__ = [
    "LANES",
    "dual_lane_convergence_record",
    "make_dual_lane_dataset",
    "read_dual_lane_contracts",
    "validate_and_resolve_dual_lane",
]
