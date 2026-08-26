"""Balanced binary/graded Full-Wh training for the dual-lane operator.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from phc2d.forward.full_cell import (
    _FullCellH5Dataset,
    _evaluate,
    _epoch_100_continue_decision,
    _convergence_record,
    _git_commit,
    _plot_history,
    _read_contract,
    _sha256_checkpoint,
    _write_history,
)
from phc2d.models import build_model, describe_model_identity, get_model_spec
from phc2d.runs import (
    RUN_LOG_NAME,
    ensure_figures_dir,
    make_run_dir,
    setup_logging,
    validate_run_artifacts,
    write_json,
    write_metrics,
    write_resolved_config,
    write_results,
    write_run_meta,
)
from phc2d.training import DeviceConfig, resolve_device


LANES = ("binary", "graded")


def _dual_lane_convergence_record(
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


def recover_fixed_200_finalizer(
    run_dir: str | Path,
    *,
    source_slurm_job_id: str | None = None,
) -> Path:
    """Complete a fixed-200 run whose training succeeded before JSON finalization."""
    out = Path(run_dir).resolve()
    missing_sources = [
        name
        for name in ("config.resolved.yaml", "history.csv", "model.pt", "epoch200.pt")
        if not (out / name).is_file()
    ]
    if missing_sources:
        raise RuntimeError(f"missing finalizer recovery sources: {missing_sources}")
    existing_outputs = [
        name
        for name in ("metrics.json", "results.json", "model_summary.json")
        if (out / name).exists()
    ]
    if existing_outputs:
        raise RuntimeError(f"refusing to overwrite existing finalizer outputs: {existing_outputs}")

    resolved_raw = yaml.safe_load((out / "config.resolved.yaml").read_text(encoding="utf-8"))
    if not isinstance(resolved_raw, Mapping):
        raise RuntimeError("resolved config must be a mapping")
    resolved = dict(resolved_raw)
    required_stop_epoch = int(resolved.get("training", {}).get("required_stop_epoch", 0))
    if required_stop_epoch != 200:
        raise RuntimeError("finalizer recovery is restricted to fixed-200 runs")
    contracts = _lane_contracts(resolved)

    with (out / "history.csv").open(encoding="utf-8", newline="") as handle:
        raw_history = [dict(row) for row in csv.DictReader(handle)]
    history: list[dict[str, Any]] = []
    integer_fields = {"epoch", "optimizer_steps", "samples_seen"}
    for raw_row in raw_history:
        history.append(
            {
                key: int(value) if key in integer_fields else float(value)
                for key, value in raw_row.items()
            }
        )
    if [row["epoch"] for row in history] != list(range(1, required_stop_epoch + 1)):
        raise RuntimeError("history.csv must cover exactly epochs 1-200")

    threshold = float(resolved["training"].get("convergence_relative_threshold", 0.005))
    convergence = _dual_lane_convergence_record(
        history,
        threshold=threshold,
        contract_metric=str(
            resolved["training"].get(
                "convergence_metric", "validation_bands_2_10_mre"
            )
        ),
    )
    json.dumps(convergence)

    import torch

    try:
        checkpoint_payload = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint_payload = torch.load(out / "model.pt", map_location="cpu")
    best_epoch = int(checkpoint_payload["epoch"])
    if not 1 <= best_epoch <= required_stop_epoch:
        raise RuntimeError(f"invalid best checkpoint epoch: {best_epoch}")
    best_row = history[best_epoch - 1]
    identity = checkpoint_payload.get("model_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("best checkpoint model identity is missing")
    model_name = str(checkpoint_payload.get("model_name", resolved["model"]["name"]))
    if model_name != str(resolved["model"]["name"]):
        raise RuntimeError("best checkpoint model name does not match resolved config")

    checkpoint = out / "model.pt"
    terminal_checkpoint = out / "epoch200.pt"
    train_time_s = float(history[-1]["wall_time_s"])
    convergence_passed = all(
        bool(item["passed"]) for item in convergence["curves"].values()
    )
    status = (
        "dual_lane_training_success"
        if convergence_passed
        else "training_complete_convergence_not_met"
    )
    metrics = {
        "validation_equal_lane_bands_2_10_mre_pct": float(best_row["val_bands_2_10_mre_pct"]),
        "binary_validation_bands_2_10_mre_pct": float(best_row["binary_val_bands_2_10_mre_pct"]),
        "graded_validation_bands_2_10_mre_pct": float(best_row["graded_val_bands_2_10_mre_pct"]),
        "best_epoch": best_epoch,
        "stop_epoch": required_stop_epoch,
        "epoch_100_continue_gate": None,
        "train_time_s": train_time_s,
        "wall_time_total_s": train_time_s,
        "parameter_count": int(identity["parameter_count"]),
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_status": "unavailable_after_failed_original_finalizer",
        "optimizer_steps": int(history[-1]["optimizer_steps"]),
        "samples_seen": int(history[-1]["samples_seen"]),
        "binary_n_train": len(contracts["binary"]["selected_train"]),
        "graded_n_train": len(contracts["graded"]["selected_train"]),
        "binary_n_val": len(contracts["binary"]["idx_val"]),
        "graded_n_val": len(contracts["graded"]["idx_val"]),
        "model_identity": dict(identity),
        "convergence": convergence,
        "required_stop_epoch": required_stop_epoch,
        "terminal_checkpoint_path": str(terminal_checkpoint),
        "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint),
        "history_epoch_count": len(history),
        "finalizer_recovery": True,
    }
    write_metrics(out, metrics)
    write_json(
        out / "model_summary.json",
        {
            "model": model_name,
            "parameter_count": metrics["parameter_count"],
            "input_shape": list(checkpoint_payload["input_shape"]),
            "output_shape": list(checkpoint_payload["output_shape"]),
            "best_validation_checkpoint": str(checkpoint),
            "best_validation_checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "terminal_checkpoint": str(terminal_checkpoint),
            "terminal_checkpoint_sha256": metrics["terminal_checkpoint_sha256"],
            "model_identity": dict(identity),
            "finalizer_recovery": True,
        },
    )
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": True,
            "mpb_meep_used": False,
            "test_executed": False,
            "model": model_name,
            "model_identity": dict(identity),
            "head_kind": str(resolved["model"]["params"].get("head_kind", "dense")),
            "seed": int(resolved["training"]["seed"]),
            "lane": "joint",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "terminal_checkpoint_path": str(terminal_checkpoint),
            "terminal_checkpoint_sha256": metrics["terminal_checkpoint_sha256"],
            "required_stop_epoch": required_stop_epoch,
            "convergence": convergence,
            "history_epoch_count": len(history),
            "best_epoch": best_epoch,
            "stop_epoch": required_stop_epoch,
            "epoch_100_continue_gate": None,
            "epoch_100_checkpoint_path": None,
            "epoch_100_checkpoint_sha256": None,
            "actual_stop_checkpoint_path": None,
            "actual_stop_checkpoint_sha256": None,
            "selection_metric": "equal_lane_validation_bands_2_10_mre_pct",
            "best_validation_equal_lane_mre_pct": metrics[
                "validation_equal_lane_bands_2_10_mre_pct"
            ],
            "next_gate": "10k_head_selection",
            "finalizer_recovery": True,
        },
    )
    write_json(
        out / "finalizer_recovery.json",
        {
            "status": "recovered",
            "reason": "original finalizer JSON circular reference",
            "source_slurm_job_id": source_slurm_job_id,
            "source_history_csv": str(out / "history.csv"),
            "source_checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "source_terminal_checkpoint_sha256": metrics["terminal_checkpoint_sha256"],
            "recovery_scope": "missing finalizer artifacts only; no training or checkpoint rewrite",
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete recovered dual-lane artifact contract: {missing}")
    return out


def _validate_and_resolve(
    config: Mapping[str, Any],
    *,
    seed: int,
    admission_probe: bool,
    head_kind: str,
) -> dict[str, Any]:
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


def _lane_contracts(resolved: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
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


def _dataset(contract: Mapping[str, Any], resolved: Mapping[str, Any], indices: Any) -> Any:
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


def run_full_cell_dual_lane(
    config_path: Path,
    *,
    seed: int,
    head_kind: str,
    run_dir: Path,
    admission_probe: bool = False,
    warm_start_run: str | Path | None = None,
) -> Path:
    """Train or probe one balanced dual-lane model checkpoint."""

    config_file = Path(config_path)
    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dual-lane config root must be a mapping")
    mutable_payload = dict(payload)
    if warm_start_run is not None:
        mutable_payload["continuation"] = {
            **dict(mutable_payload.get("continuation") or {}),
            "source_run": str(Path(warm_start_run).resolve()),
        }
    resolved = _validate_and_resolve(
        mutable_payload,
        seed=seed,
        admission_probe=admission_probe,
        head_kind=head_kind,
    )
    out = make_run_dir(
        str(resolved["experiment"]["experiment_id"]),
        runs_dir=resolved["output"].get("runs_dir", "runs"),
        run_dir=run_dir,
    )
    logger = setup_logging(
        log_file=out / RUN_LOG_NAME,
        run_meta={
            "script": "python -m phc2d.forward.full_cell_dual_lane",
            "config_path": str(config_file),
            "model": resolved["model"]["name"],
            "head_kind": head_kind,
            "seed": int(seed),
            "validation_boundary": "dual_lane_admission_probe" if admission_probe else "dual_lane_training",
        },
    )
    write_resolved_config(out, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    ensure_figures_dir(out)

    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_info = resolve_device(DeviceConfig.from_mapping(resolved["runtime"]), torch_module=torch)
    device = device_info.torch_device
    contracts = _lane_contracts(resolved)
    spec = get_model_spec(str(resolved["model"]["name"]))
    reference = contracts["binary"]
    model = build_model(
        spec.name,
        input_shape=reference["grid_shape"],
        output_shape=reference["target_shape"],
        k_points=reference["k_points"],
        **dict(resolved["model"]["params"]),
    ).to(device)
    warm_start: dict[str, Any] | None = None
    if resolved.get("continuation"):
        from phc2d.forward.full_cell_warm_start import load_weight_only_warm_start

        source_run = resolved["continuation"].get("source_run")
        if not source_run:
            raise ValueError("dual-lane continuation source_run is required")
        warm_start = load_weight_only_warm_start(
            source_run,
            expected_model=spec.name,
            expected_seed=seed,
            torch_module=torch,
            expected_source_experiment=resolved["continuation"].get("source_experiment"),
        )
        model.load_state_dict(warm_start["terminal_payload"]["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(resolved["training"]["learning_rate"]),
        weight_decay=float(resolved["training"].get("weight_decay", 1.0e-5)),
    )
    loss_fn = torch.nn.MSELoss()
    half_batch = int(resolved["training"]["micro_batch_size"]) // 2
    loader_kwargs = {
        "batch_size": half_batch,
        "num_workers": int(device_info.num_workers),
        "pin_memory": bool(device_info.pin_memory),
    }
    train_loaders = {
        lane: DataLoader(
            _dataset(contracts[lane], resolved, contracts[lane]["selected_train"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + lane_index),
            **loader_kwargs,
        )
        for lane_index, lane in enumerate(LANES)
    }
    val_datasets = {
        lane: _dataset(contracts[lane], resolved, contracts[lane]["idx_val"])
        for lane in LANES
    }
    val_loaders = {
        lane: DataLoader(val_datasets[lane], shuffle=False, **loader_kwargs)
        for lane in LANES
    }
    means = {
        lane: torch.as_tensor(contracts[lane]["target_mean"], dtype=torch.float32, device=device)
        for lane in LANES
    }
    stds = {
        lane: torch.as_tensor(contracts[lane]["target_std"], dtype=torch.float32, device=device)
        for lane in LANES
    }
    k_points = torch.as_tensor(reference["k_points"], dtype=torch.float32, device=device)

    history: list[dict[str, Any]] = [] if warm_start is None else list(warm_start["source_history"])
    best_score = float("inf") if warm_start is None else min(float(row["val_bands_2_10_mre_pct"]) for row in history)
    best_epoch = 0 if warm_start is None else int(warm_start["source_result"]["best_epoch"])
    stale_epochs = 0
    optimizer_steps = 0 if warm_start is None else int(history[-1]["optimizer_steps"])
    samples_seen = 0 if warm_start is None else int(history[-1]["samples_seen"])
    wall_time_offset = 0.0 if warm_start is None else float(history[-1]["wall_time_s"])
    continuation_convergence: dict[str, Any] | None = None
    if warm_start is not None:
        from phc2d.forward.full_cell_warm_start import seed_derivative_best_checkpoint

        seed_derivative_best_checkpoint(warm_start, out)
    wall_start = time.perf_counter()
    train_start = time.perf_counter()
    gate_enabled = bool(resolved["training"].get("epoch_100_continue_gate", False)) and not admission_probe
    gate_record: dict[str, Any] | None = None
    epoch100_snapshot: Path | None = None
    start_epoch = 1 if warm_start is None else int(warm_start["source_epoch"]) + 1
    for epoch in range(start_epoch, int(resolved["training"]["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_sum = 0.0
        train_count = 0
        for batch_index, ((xb, yb), (xg, yg)) in enumerate(
            zip(train_loaders["binary"], train_loaders["graded"], strict=True),
            start=1,
        ):
            inputs = torch.cat((xb, xg), dim=0).to(device, non_blocking=True)
            targets = torch.cat((yb, yg), dim=0).to(device, non_blocking=True)
            lane_id = torch.cat(
                (
                    torch.zeros(xb.shape[0], dtype=torch.long),
                    torch.ones(xg.shape[0], dtype=torch.long),
                )
            ).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, k_points=k_points, lane_id=lane_id)
            loss = loss_fn(prediction, targets)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"non-finite dual-lane loss at epoch {epoch}, batch {batch_index}")
            loss.backward()
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError(f"non-finite dual-lane gradient at epoch {epoch}, batch {batch_index}")
            optimizer.step()
            optimizer_steps += 1
            batch_count = int(inputs.shape[0])
            train_sum += float(loss.detach().cpu()) * batch_count
            train_count += batch_count
            samples_seen += batch_count
            if resolved["training"].get("max_train_batches") is not None:
                break
        evaluations = {
            lane: _evaluate(
                model,
                val_loaders[lane],
                device=device,
                mean=means[lane],
                std=stds[lane],
                model_spec=spec,
                k_points=k_points,
                lane_index=lane_index,
                torch=torch,
            )
            for lane_index, lane in enumerate(LANES)
        }
        score = float(np.mean([evaluations[lane]["bands_2_10_mre_pct"] for lane in LANES]))
        row = {
            "epoch": epoch,
            "train_loss": train_sum / max(train_count, 1),
            "val_loss": float(np.mean([evaluations[lane]["normalized_mse"] for lane in LANES])),
            "val_band1_mre_pct": float(np.mean([evaluations[lane]["band1_mre_pct"] for lane in LANES])),
            "val_bands_2_10_mre_pct": score,
            "val_bands_1_10_mre_pct": float(np.mean([evaluations[lane]["bands_1_10_mre_pct"] for lane in LANES])),
            "binary_val_bands_2_10_mre_pct": evaluations["binary"]["bands_2_10_mre_pct"],
            "graded_val_bands_2_10_mre_pct": evaluations["graded"]["bands_2_10_mre_pct"],
            "learning_rate": float(resolved["training"]["learning_rate"]),
            "epoch_time_s": time.perf_counter() - epoch_start,
            "wall_time_s": wall_time_offset + time.perf_counter() - wall_start,
            "optimizer_steps": optimizer_steps,
            "samples_seen": samples_seen,
        }
        history.append(row)
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_tmp = out / "model.pt.tmp"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": spec.name,
                    "model_params": dict(resolved["model"]["params"]),
                    "model_identity": describe_model_identity(
                        model, spec.name, spec, model_kwargs=resolved["model"]["params"]
                    ),
                    "input_shape": reference["grid_shape"],
                    "output_shape": reference["target_shape"],
                    "k_points": torch.as_tensor(reference["k_points"], dtype=torch.float32),
                    "k_points_sha256": reference["k_points_sha256"],
                    "lane_mapping": {"binary": 0, "graded": 1},
                    "target_normalizers": {
                        lane: {
                            "mean": torch.as_tensor(contracts[lane]["target_mean"]),
                            "std": torch.as_tensor(contracts[lane]["target_std"]),
                        }
                        for lane in LANES
                    },
                    "epoch": epoch,
                    "validation_equal_lane_mre_pct": best_score,
                    "resolved_config": resolved,
                    "warm_start": None if warm_start is None else {
                        key: value for key, value in warm_start.items()
                        if key not in {"source_history", "source_result", "terminal_payload"}
                    },
                },
                checkpoint_tmp,
            )
            os.replace(checkpoint_tmp, out / "model.pt")
        else:
            stale_epochs += 1
        _write_history(out / "history.csv", history)
        logger.info(
            "epoch=%d train_loss=%.6g binary_mre=%.6g graded_mre=%.6g equal_lane_mre=%.6g",
            epoch,
            row["train_loss"],
            row["binary_val_bands_2_10_mre_pct"],
            row["graded_val_bands_2_10_mre_pct"],
            score,
        )
        if warm_start is not None and epoch % int(resolved["continuation"]["check_interval"]) == 0:
            from phc2d.forward.full_cell_warm_start import rolling_convergence_record

            curves = {
                "binary": rolling_convergence_record(history, end_epoch=epoch, metric="binary_val_bands_2_10_mre_pct", threshold=float(resolved["continuation"]["relative_threshold"])),
                "graded": rolling_convergence_record(history, end_epoch=epoch, metric="graded_val_bands_2_10_mre_pct", threshold=float(resolved["continuation"]["relative_threshold"])),
                "equal_lane": rolling_convergence_record(history, end_epoch=epoch, metric="val_bands_2_10_mre_pct", threshold=float(resolved["continuation"]["relative_threshold"])),
            }
            continuation_convergence = {
                **curves["equal_lane"],
                "contract_metric": "validation_bands_2_10_mre",
                "curves": curves,
            }
            logger.info(
                "dual-lane continuation convergence epoch=%d binary=%s graded=%s equal_lane=%s",
                epoch,
                curves["binary"]["passed"],
                curves["graded"]["passed"],
                curves["equal_lane"]["passed"],
            )
            if bool(resolved["continuation"].get("stop_when_converged", True)) and all(
                bool(item["passed"]) for item in curves.values()
            ):
                break
        if gate_enabled and epoch == 100:
            gate_record = _epoch_100_continue_decision(
                history,
                threshold=float(resolved["training"].get("continue_gate_min_relative_improvement", 0.005)),
            )
            epoch100_snapshot = out / "epoch100.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": 100,
                    "checkpoint_role": "epoch_100_gate_snapshot",
                    "gate_record": gate_record,
                },
                epoch100_snapshot,
            )
            logger.info(
                "epoch-100 gate relative_improvement=%.8f threshold=%.8f decision=%s",
                gate_record["relative_improvement"],
                gate_record["threshold"],
                gate_record["decision"],
            )
            if gate_record["decision"] == "stop_at_100":
                break
            stale_epochs = 0
        patience = int(resolved["training"].get("early_stopping_patience", 15))
        patience_active = not gate_enabled or (epoch > 100 and gate_record is not None)
        if not admission_probe and patience_active and patience > 0 and stale_epochs >= patience:
            break

    train_time = time.perf_counter() - train_start
    stop_epoch = int(history[-1]["epoch"])
    stop_snapshot = epoch100_snapshot
    if gate_enabled and stop_epoch > 100:
        stop_snapshot = out / "continuation_stop.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": stop_epoch,
                "checkpoint_role": "post_gate_actual_stop_snapshot",
                "gate_record": gate_record,
            },
            stop_snapshot,
        )
    required_stop_epoch = resolved["training"].get("required_stop_epoch")
    convergence: dict[str, Any] | None = None
    terminal_checkpoint: Path | None = None
    if required_stop_epoch is not None and not admission_probe:
        required_stop_epoch = int(required_stop_epoch)
        if stop_epoch != required_stop_epoch:
            raise RuntimeError(f"required terminal epoch {required_stop_epoch}, observed {stop_epoch}")
        convergence_window = tuple(int(value) for value in resolved["training"].get("convergence_window", [181, 200]))
        if convergence_window != (181, 200):
            raise ValueError("EXP-FWD-008 convergence_window must be [181, 200]")
        convergence = _dual_lane_convergence_record(
            history,
            threshold=float(resolved["training"].get("convergence_relative_threshold", 0.005)),
            contract_metric=str(
                resolved["training"].get(
                    "convergence_metric", "validation_bands_2_10_mre"
                )
            ),
        )
        terminal_checkpoint = out / f"epoch{required_stop_epoch}.pt"
        terminal_tmp = terminal_checkpoint.with_suffix(terminal_checkpoint.suffix + ".tmp")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": spec.name,
                "model_params": dict(resolved["model"]["params"]),
                "model_identity": describe_model_identity(model, spec.name, spec, model_kwargs=resolved["model"]["params"]),
                "input_shape": reference["grid_shape"],
                "output_shape": reference["target_shape"],
                "k_points": torch.as_tensor(reference["k_points"], dtype=torch.float32),
                "k_points_sha256": reference["k_points_sha256"],
                "lane_mapping": {"binary": 0, "graded": 1},
                "target_normalizers": {lane: {"mean": torch.as_tensor(contracts[lane]["target_mean"]), "std": torch.as_tensor(contracts[lane]["target_std"])} for lane in LANES},
                "epoch": required_stop_epoch,
                "checkpoint_role": "terminal_epoch",
                "selection_split": "terminal_training_state",
                "convergence": convergence,
                "resolved_config": resolved,
            },
            terminal_tmp,
        )
        os.replace(terminal_tmp, terminal_checkpoint)
    elif warm_start is not None and not admission_probe:
        convergence = continuation_convergence
        if convergence is None:
            raise RuntimeError("dual-lane continuation ended without a convergence checkpoint")
        terminal_checkpoint = out / f"epoch{stop_epoch}.pt"
        terminal_tmp = terminal_checkpoint.with_suffix(terminal_checkpoint.suffix + ".tmp")
        from phc2d.forward.full_cell_warm_start import public_warm_start_metadata

        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": spec.name,
                "model_params": dict(resolved["model"]["params"]),
                "model_identity": describe_model_identity(model, spec.name, spec, model_kwargs=resolved["model"]["params"]),
                "input_shape": reference["grid_shape"],
                "output_shape": reference["target_shape"],
                "k_points": torch.as_tensor(reference["k_points"], dtype=torch.float32),
                "k_points_sha256": reference["k_points_sha256"],
                "lane_mapping": {"binary": 0, "graded": 1},
                "target_normalizers": {lane: {"mean": torch.as_tensor(contracts[lane]["target_mean"]), "std": torch.as_tensor(contracts[lane]["target_std"])} for lane in LANES},
                "epoch": stop_epoch,
                "checkpoint_role": "continuation_terminal_epoch",
                "selection_split": "terminal_training_state",
                "convergence": convergence,
                "warm_start": public_warm_start_metadata(warm_start),
                "resolved_config": resolved,
            },
            terminal_tmp,
        )
        os.replace(terminal_tmp, terminal_checkpoint)
    checkpoint = out / "model.pt"
    if not checkpoint.is_file():
        raise RuntimeError("best-validation dual-lane checkpoint was not written")
    try:
        checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint_payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    final_evaluations = {
        lane: _evaluate(
            model,
            val_loaders[lane],
            device=device,
            mean=means[lane],
            std=stds[lane],
            model_spec=spec,
            k_points=k_points,
            lane_index=lane_index,
            torch=torch,
        )
        for lane_index, lane in enumerate(LANES)
    }
    final_score = float(
        np.mean([final_evaluations[lane]["bands_2_10_mre_pct"] for lane in LANES])
    )
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device_info.is_cuda else 0.0
    metrics = {
        "validation_equal_lane_bands_2_10_mre_pct": final_score,
        "binary_validation_bands_2_10_mre_pct": final_evaluations["binary"]["bands_2_10_mre_pct"],
        "graded_validation_bands_2_10_mre_pct": final_evaluations["graded"]["bands_2_10_mre_pct"],
        "best_epoch": int(checkpoint_payload["epoch"]),
        "stop_epoch": stop_epoch,
        "epoch_100_continue_gate": gate_record,
        "train_time_s": train_time,
        "wall_time_total_s": time.perf_counter() - wall_start,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_gpu_memory_mb": peak_mb,
        "optimizer_steps": optimizer_steps,
        "samples_seen": samples_seen,
        "binary_n_train": len(train_loaders["binary"].dataset),
        "graded_n_train": len(train_loaders["graded"].dataset),
        "binary_n_val": len(val_datasets["binary"]),
        "graded_n_val": len(val_datasets["graded"]),
        "model_identity": describe_model_identity(
            model, spec.name, spec, model_kwargs=resolved["model"]["params"]
        ),
        "convergence": convergence,
        "required_stop_epoch": required_stop_epoch,
        "terminal_checkpoint_path": str(terminal_checkpoint) if terminal_checkpoint else None,
        "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
        "history_epoch_count": len(history),
        "warm_start": None if warm_start is None else public_warm_start_metadata(warm_start),
    }
    _plot_history(out, history, monitor_label="equal-lane validation")
    write_metrics(out, metrics)
    (out / "model_summary.json").write_text(
        json.dumps(
            {
                "model": spec.name,
                "parameter_count": metrics["parameter_count"],
                "input_shape": list(reference["grid_shape"]),
                "output_shape": list(reference["target_shape"]),
                "best_validation_checkpoint": str(checkpoint),
                "best_validation_checkpoint_sha256": _sha256_checkpoint(checkpoint),
                "terminal_checkpoint": str(terminal_checkpoint) if terminal_checkpoint else None,
                "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
                "model_identity": metrics["model_identity"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = (
        "dual_lane_admission_probe_pass"
        if admission_probe
        else "continuation_training_success"
        if warm_start is not None
        else "dual_lane_training_success"
    )
    if not admission_probe and convergence is not None and not all(bool(item["passed"]) for item in convergence["curves"].values()):
        status = "training_complete_convergence_not_met"
    write_results(
        out,
        {
            "status": status,
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
            "test_executed": False,
            "model": spec.name,
            "model_identity": metrics["model_identity"],
            "head_kind": head_kind,
            "seed": int(seed),
            "lane": "joint",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256_checkpoint(checkpoint),
            "terminal_checkpoint_path": str(terminal_checkpoint) if terminal_checkpoint else None,
            "terminal_checkpoint_sha256": _sha256_checkpoint(terminal_checkpoint) if terminal_checkpoint else None,
            "required_stop_epoch": required_stop_epoch,
            "convergence": convergence,
            "history_epoch_count": len(history),
            "warm_start": None if warm_start is None else public_warm_start_metadata(warm_start),
            "best_epoch": int(checkpoint_payload["epoch"]),
            "stop_epoch": stop_epoch,
            "epoch_100_continue_gate": gate_record,
            "epoch_100_checkpoint_path": str(epoch100_snapshot) if epoch100_snapshot else None,
            "epoch_100_checkpoint_sha256": _sha256_checkpoint(epoch100_snapshot) if epoch100_snapshot else None,
            "actual_stop_checkpoint_path": str(stop_snapshot) if stop_snapshot else None,
            "actual_stop_checkpoint_sha256": _sha256_checkpoint(stop_snapshot) if stop_snapshot else None,
            "selection_metric": "equal_lane_validation_bands_2_10_mre_pct",
            "next_gate": "dual_lane_admission_decision" if admission_probe else "continuation_checkpoint_freeze" if warm_start is not None else "10k_head_selection",
        },
    )
    write_run_meta(
        out,
        {
            "script": "python -m phc2d.forward.full_cell_dual_lane",
            "config_path": str(config_file),
            "deployed_git_commit": _git_commit(),
            "dataset_paths": {lane: str(contracts[lane]["path"]) for lane in LANES},
            "dataset_sha256": {lane: contracts[lane]["dataset_sha256"] for lane in LANES},
            "idx_train_hash": {lane: contracts[lane]["idx_train_hash"] for lane in LANES},
            "idx_val_hash": {lane: contracts[lane]["idx_val_hash"] for lane in LANES},
            "idx_test_hash": {lane: contracts[lane]["idx_test_hash"] for lane in LANES},
            "k_points_sha256": reference["k_points_sha256"],
            "model": spec.name,
            "head_kind": head_kind,
            "seed": int(seed),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_node": os.environ.get("SLURM_JOB_NODELIST"),
            **device_info.metadata(peak_gpu_memory_mb=peak_mb),
        },
    )
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete dual-lane run artifact contract: {missing}")
    logger.info("terminal_status=%s run_dir=%s", status, out)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--head-kind", choices=("dense", "k_conditioned"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--admission-probe", action="store_true")
    parser.add_argument("--warm-start-run", type=Path)
    args = parser.parse_args(argv)
    print(
        run_full_cell_dual_lane(
            args.config,
            seed=args.seed,
            head_kind=args.head_kind,
            run_dir=args.run_dir,
            admission_probe=args.admission_probe,
            warm_start_run=args.warm_start_run,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["recover_fixed_200_finalizer", "run_full_cell_dual_lane"]
