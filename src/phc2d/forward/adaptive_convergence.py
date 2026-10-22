"""Validation-only adaptive optimization and restart evidence for Full-Wh training.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def validate_adaptive_contract(config: dict, experiment_id: str, training: dict, continuation: dict) -> None:
    """Keep adaptive optimization separate from immutable fixed-budget experiments."""
    if experiment_id not in {"EXP-FWD-026", "EXP-FWD-027", "EXP-FWD-028"}:
        raise ValueError("adaptive optimization requires EXP-FWD-026/027/028")
    expected = dict(start_epoch=300, minimum_extension=100, check_interval=20,
                    improvement_window=80, improvement_threshold=0.001,
                    stability_window=60, stability_threshold=0.005, confirmations=2,
                    initial_lr_factor=0.2, min_lr_factor=0.01, factor=0.5,
                    patience=30, cooldown=10)
    if config != expected:
        raise ValueError("adaptive contract differs from the frozen convergence protocol")
    if int(training["max_epochs"]) != 1000 or training.get("required_stop_epoch") is not None:
        raise ValueError("adaptive budget must cap at epoch 1000 without a fixed stop")
    if int(training.get("early_stopping_patience", -1)) != 0 or training.get("epoch_100_continue_gate"):
        raise ValueError("adaptive optimization owns the stopping decision")
    if continuation:
        if experiment_id not in {"EXP-FWD-026", "EXP-FWD-027"}:
            raise ValueError("other-model comparison must start from scratch")
        if continuation.get("mode") != "epoch300_best_weights_optimizer_reset" or int(continuation.get("source_epoch", -1)) != 300:
            raise ValueError("adaptive warm start requires frozen epoch-300 validation-best weights")
    elif experiment_id != "EXP-FWD-028":
        raise ValueError("core and branch extensions require a source run")


def convergence_record(history: list[dict], config: dict, *, floor_since: int | None, streak: int) -> dict:
    """Require two stable validation windows after a sustained final learning rate."""
    epoch = int(history[-1]["epoch"])
    start = int(config["start_epoch"])
    by_epoch = {int(r["epoch"]): r for r in history}
    if epoch - start < config["minimum_extension"]:
        return dict(passed=False, checked_at_epoch=epoch, streak=0, reason="minimum_extension")
    recent = [by_epoch[e] for e in range(epoch - config["improvement_window"], epoch + 1)]
    spreads = {}
    stable = True
    for metric in ("val_bands_2_10_mre_pct", "val_bands_1_10_mre_pct"):
        values = [float(r[metric]) for r in recent]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("non-finite or negative convergence metric")
        tail = values[-config["stability_window"]:]
        width = len(tail) // 3
        means = [sum(tail[i:i + width]) / width for i in range(0, len(tail), width)]
        spread = (max(means) - min(means)) / max(abs(sum(means) / 3), 1e-12)
        incumbent = min(float(r[metric]) for r in history if int(r["epoch"]) <= epoch - config["improvement_window"])
        improvement = max(0.0, (incumbent - min(values)) / max(abs(incumbent), 1e-12))
        # A stable but substantially overfit tail is not convergence of the selected model.
        best = min(float(r[metric]) for r in history)
        guard = means[-1] <= best * 1.1 + 1e-12
        spreads[metric] = dict(window_means=means, relative_range=spread,
                               relative_improvement=improvement, selected_model_guard=guard)
        stable &= spread <= config["stability_threshold"] and improvement < config["improvement_threshold"] and guard
    floor_ready = floor_since is not None and epoch - floor_since >= config["improvement_window"]
    streak = streak + 1 if stable and floor_ready else 0
    return dict(passed=streak >= config["confirmations"], checked_at_epoch=epoch,
                streak=streak, floor_since=floor_since, floor_ready=floor_ready,
                curves=spreads, reason="validation_plateau" if streak >= config["confirmations"] else "not_converged")


class AdaptiveOptimization:
    """Own scheduling, convergence checks and complete restart snapshots, not training."""

    def __init__(self, optimizer: Any, config: dict, base_lr: float, torch_module: Any) -> None:
        self.optimizer, self.config, self.torch = optimizer, config, torch_module
        self.base_lr = float(base_lr)
        self.scheduler = None
        self.floor_since = None
        self.streak = 0
        self.record = dict(passed=False, reason="fixed_prefix")

    def before_epoch(self, epoch: int) -> None:
        if epoch == self.config["start_epoch"] + 1:
            for group in self.optimizer.param_groups:
                group["lr"] = self.base_lr * self.config["initial_lr_factor"]
            self.scheduler = self.torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=self.config["factor"],
                patience=self.config["patience"], threshold=self.config["improvement_threshold"],
                threshold_mode="rel", cooldown=self.config["cooldown"],
                min_lr=self.base_lr * self.config["min_lr_factor"], eps=0.0)

    def after_epoch(self, history: list[dict]) -> bool:
        epoch = int(history[-1]["epoch"])
        if self.scheduler is not None:
            self.scheduler.step(float(history[-1]["val_bands_2_10_mre_pct"]))
            if self.optimizer.param_groups[0]["lr"] <= self.base_lr * self.config["min_lr_factor"] * (1 + 1e-9):
                if self.floor_since is None:
                    self.floor_since = epoch
            if epoch % self.config["check_interval"] == 0:
                self.record = convergence_record(history, self.config, floor_since=self.floor_since, streak=self.streak)
                self.streak = self.record["streak"]
        return bool(self.record["passed"])

    def save(self, path: Path, model: Any, generator: Any, history: list[dict], resolved: dict) -> None:
        numpy_state = np.random.get_state()
        payload = dict(state_dict=model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(),
                       scheduler_state_dict=None if self.scheduler is None else self.scheduler.state_dict(),
                       python_rng_state=random.getstate(),
                       numpy_rng_state=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
                       torch_rng_state=self.torch.get_rng_state(),
                       cuda_rng_states=self.torch.cuda.get_rng_state_all() if self.torch.cuda.is_available() else [],
                       data_order_generator_state=generator.get_state(), epoch=history[-1]["epoch"],
                       controller_state=dict(floor_since=self.floor_since, streak=self.streak, record=self.record),
                       history=history, resolved_config=resolved, checkpoint_role="full_training_state")
        temporary = path.with_suffix(path.suffix + ".tmp")
        self.torch.save(payload, temporary)
        os.replace(temporary, path)
        (Path(path).parent / "adaptive_progress.json").write_text(json.dumps(dict(
            epoch=history[-1]["epoch"], learning_rate=self.optimizer.param_groups[0]["lr"],
            convergence=self.record, restart_checkpoint=str(path)), indent=2) + "\n", encoding="utf-8")
