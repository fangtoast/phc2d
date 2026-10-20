"""Configuration validation for formal Full-Wh training contracts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations
from typing import Any
from phc2d.models import get_model_spec

def resolve_full_cell_training_config(
    config: dict[str, Any],
    *,
    model_name: str,
    seed: int,
    admission_probe: bool,
    threshold_override: float | None = None,
    resolution_override: int | None = None,
    train_size_override: int | None = None,
    max_epochs_override: int | None = None,
    head_kind_override: str | None = None,
) -> dict[str, Any]:
    dataset = dict(config.get("dataset") or {})
    representation = dict(config.get("representation") or {})
    if threshold_override is not None:
        representation["threshold"] = float(threshold_override)
    if resolution_override is not None:
        if int(resolution_override) not in {32, 64, 128}:
            raise ValueError("resolution override must be 32, 64, or 128")
        representation["resolution"] = int(resolution_override)
        dataset["grid_field"] = f"material_weight_grid_{int(resolution_override)}"
    models = dict(config.get("models") or {})
    training = dict(config.get("training") or {})
    continuation = dict(config.get("continuation") or {})
    runtime = dict(config.get("runtime") or {})
    experiment = dict(config.get("experiment") or {})
    output = dict(config.get("output") or {})

    if model_name not in models:
        raise ValueError(f"model {model_name!r} is not configured under models")
    model_spec = get_model_spec(model_name)
    if model_spec.output_target != "Wh":
        raise ValueError(f"model {model_name!r} does not produce the Wh target")
    if int(seed) not in {0, 1, 2}:
        raise ValueError("formal full-cell forward seeds are fixed to 0, 1, 2")
    sampling = str(representation.get("sampling", "continuous"))
    threshold = representation.get("threshold")
    if sampling not in {"continuous", "deterministic_threshold"}:
        raise ValueError("representation.sampling must be continuous or deterministic_threshold")
    if sampling == "deterministic_threshold":
        if threshold not in {0.3, 0.4, 0.5, 0.6, 0.7}:
            raise ValueError("threshold must be one of 0.3, 0.4, 0.5, 0.6, 0.7")
        if bool(representation.get("physical_equivalence", True)):
            raise ValueError("threshold proxy inputs must set physical_equivalence=false")
    elif threshold is not None:
        raise ValueError("continuous representation requires threshold=null")

    layout = model_spec.input_layout
    learning_rates = dict(training.get("learning_rates") or {})
    if model_name not in learning_rates:
        raise ValueError(f"training.learning_rates is missing {model_name!r}")
    if train_size_override is not None:
        training["train_size"] = int(train_size_override)
    if max_epochs_override is not None:
        training["max_epochs"] = int(max_epochs_override)
    train_size = training.get("train_size")
    if train_size is not None and int(train_size) <= 0:
        raise ValueError("training.train_size must be positive")
    max_epochs = int(training.get("max_epochs", 100))
    if max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive")
    experiment_id = str(experiment.get("experiment_id", "full_cell_forward"))
    if sampling == "deterministic_threshold":
        experiment_id = f"{experiment_id}_t{int(round(float(threshold) * 10)):02d}"
    resolution = int(representation.get("resolution", 64))
    if resolution != 64:
        experiment_id = f"{experiment_id}_r{resolution}"
    if train_size is not None:
        experiment_id = f"{experiment_id}_n{int(train_size)}_e{max_epochs}"
    model_params = dict(models.get(model_name) or {})
    if head_kind_override is not None:
        if head_kind_override not in {"dense", "k_conditioned"}:
            raise ValueError("head kind override must be dense or k_conditioned")
        if not model_spec.accepts_k_points:
            raise ValueError(f"model {model_name!r} does not support selectable Full-Wh heads")
        model_params["head_kind"] = head_kind_override
    resolved = {
        "experiment": {
            **experiment,
            "experiment_id": experiment_id,
            "model": model_name,
            "seed": int(seed),
            "admission_probe": bool(admission_probe),
            "formal_training_started": not admission_probe,
            "mpb_meep_used": False,
        },
        "dataset": dataset,
        "representation": {**representation, "layout": layout, "flatten_order": "row_major"},
        "model": {"name": model_name, "params": model_params},
        "training": {
            **training,
            "seed": int(seed),
            "learning_rate": float(learning_rates[model_name]),
            # A continuation probe must exercise the first resumed epoch, rather
            # than silently becoming an epoch-1 probe that never enters its loop.
            "max_epochs": (
                int(continuation.get("source_epoch", 200)) + 1
                if admission_probe and continuation
                else 1
                if admission_probe
                else max_epochs
            ),
            "max_train_batches": 1 if admission_probe else None,
        },
        "runtime": runtime,
        "output": output,
        "continuation": continuation,
    }
    if not admission_probe:
        continuation_enabled = bool(continuation)
        allowed_epochs = {20, 100, 200, 300, 600} if continuation_enabled else {20, 100, 200}
        if str(experiment.get("experiment_id")) in {"EXP-FWD-024", "EXP-FWD-025"}:
            allowed_epochs.add(300)
            if continuation_enabled or max_epochs != 300:
                raise ValueError("fixed-300 confirmation requires from-scratch epoch 300")
            if int(training.get("required_stop_epoch", -1)) != 300:
                raise ValueError("fixed-300 confirmation requires terminal epoch 300")
        if int(resolved["training"]["max_epochs"]) not in allowed_epochs:
            raise ValueError(f"formal training max_epochs must be one of {sorted(allowed_epochs)}")
        if int(training.get("early_stopping_patience", 0)) < 0:
            raise ValueError("early_stopping_patience must be nonnegative")
        if int(training.get("effective_batch_size", 0)) != 64:
            raise ValueError("formal training requires effective_batch_size=64")
        if str(training.get("optimizer", "")).lower() != "adamw":
            raise ValueError("formal training requires AdamW")
        if str(runtime.get("precision", "")).lower() != "fp32":
            raise ValueError("formal training requires FP32")
        if bool(runtime.get("amp_enabled")) or bool(runtime.get("tf32_enabled")):
            raise ValueError("formal training forbids AMP and TF32")
        if continuation_enabled:
            continuation_experiment = str(experiment.get("experiment_id"))
            if continuation_experiment not in {"EXP-FWD-009", "EXP-FWD-020"}:
                raise ValueError("weight-only continuation requires a registered experiment")
            if str(continuation.get("mode")) != "epoch200_weights_optimizer_reset":
                raise ValueError("unsupported continuation mode")
            if int(continuation.get("source_epoch", -1)) != 200:
                raise ValueError("continuation source_epoch must equal 200")
            if int(continuation.get("check_interval", -1)) != 20:
                raise ValueError("continuation check_interval must equal 20")
            if float(continuation.get("relative_threshold", -1.0)) != 0.005:
                raise ValueError("continuation relative_threshold must equal 0.005")
            if not bool(continuation.get("optimizer_reset", False)):
                raise ValueError("weight-only continuation must declare optimizer_reset=true")
            if continuation_experiment == "EXP-FWD-009":
                if int(resolved["training"]["max_epochs"]) != 600:
                    raise ValueError("EXP-FWD-009 continuation max_epochs must equal 600")
                if training.get("required_stop_epoch") is not None:
                    raise ValueError("continuation uses a convergence stop, not required_stop_epoch")
            else:
                if int(resolved["training"]["max_epochs"]) != 300 or int(continuation.get("max_total_epoch", -1)) != 300:
                    raise ValueError("EXP-FWD-020 continuation must run through total epoch 300")
                if bool(continuation.get("stop_when_converged", True)):
                    raise ValueError("EXP-FWD-020 continuation forbids early convergence stop")
                if int(training.get("required_stop_epoch", -1)) != 300:
                    raise ValueError("EXP-FWD-020 continuation requires terminal epoch 300")
            if int(training.get("early_stopping_patience", -1)) != 0:
                raise ValueError("continuation requires early_stopping_patience=0")
        if training.get("required_stop_epoch") is not None:
            required_stop = int(training["required_stop_epoch"])
            allowed_required_stops = {200}
            if str(experiment.get("experiment_id")) in {"EXP-FWD-024", "EXP-FWD-025"}:
                allowed_required_stops.add(300)
            if continuation_enabled and str(experiment.get("experiment_id")) == "EXP-FWD-020":
                allowed_required_stops.add(300)
            if max_epochs != required_stop or required_stop not in allowed_required_stops:
                raise ValueError("fixed training requires matching registered terminal epoch")
            if int(training.get("early_stopping_patience", -1)) != 0:
                raise ValueError("fixed-200 training requires early_stopping_patience=0")
            if bool(training.get("epoch_100_continue_gate", False)):
                raise ValueError("fixed-200 training forbids epoch_100_continue_gate")
            if not bool(training.get("require_terminal_checkpoint", False)):
                raise ValueError("fixed-200 training requires a terminal checkpoint")
            expected_window = (281, 300) if required_stop == 300 else (181, 200)
            if tuple(int(value) for value in training.get("convergence_window", [])) != expected_window:
                raise ValueError(f"fixed training requires convergence_window={list(expected_window)}")
            if str(training.get("convergence_metric", "")) != "validation_bands_2_10_mre":
                raise ValueError("fixed-200 training requires validation_bands_2_10_mre")
            if float(training.get("convergence_relative_threshold", -1.0)) != 0.005:
                raise ValueError("fixed-200 convergence threshold must be 0.005")
        gate_enabled = bool(training.get("epoch_100_continue_gate", False))
        if gate_enabled:
            if int(resolved["training"]["max_epochs"]) != 200:
                raise ValueError("epoch-100 continuation gate requires max_epochs=200")
            threshold = float(training.get("continue_gate_min_relative_improvement", 0.005))
            if threshold != 0.005:
                raise ValueError("preregistered continuation threshold must equal 0.005")
            if int(training.get("early_stopping_patience", 0)) != 15:
                raise ValueError("epoch-100 continuation requires early_stopping_patience=15")
    return resolved
