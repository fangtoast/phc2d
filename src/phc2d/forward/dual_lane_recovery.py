"""Artifact-only recovery for completed fixed-200 dual-lane training runs.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

import yaml

from phc2d.forward.dual_lane_contracts import (
    dual_lane_convergence_record,
    read_dual_lane_contracts,
)
from phc2d.forward.full_cell import _sha256_checkpoint
from phc2d.runs import (
    validate_run_artifacts,
    write_json,
    write_metrics,
    write_results,
)


def recover_fixed_200_finalizer(
    run_dir: str | Path,
    *,
    source_slurm_job_id: str | None = None,
    contract_reader: Callable[[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]] = read_dual_lane_contracts,
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
    contracts = contract_reader(resolved)
    with (out / "history.csv").open(encoding="utf-8", newline="") as handle:
        raw_history = [dict(row) for row in csv.DictReader(handle)]
    integer_fields = {"epoch", "optimizer_steps", "samples_seen"}
    history: list[dict[str, Any]] = [
        {key: int(value) if key in integer_fields else float(value) for key, value in row.items()}
        for row in raw_history
    ]
    if [row["epoch"] for row in history] != list(range(1, required_stop_epoch + 1)):
        raise RuntimeError("history.csv must cover exactly epochs 1-200")
    convergence = dual_lane_convergence_record(
        history,
        threshold=float(resolved["training"].get("convergence_relative_threshold", 0.005)),
        contract_metric=str(resolved["training"].get("convergence_metric", "validation_bands_2_10_mre")),
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
    status = "dual_lane_training_success" if all(bool(item["passed"]) for item in convergence["curves"].values()) else "training_complete_convergence_not_met"
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
    write_json(out / "model_summary.json", {
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
    })
    write_results(out, {
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
        "best_validation_equal_lane_mre_pct": metrics["validation_equal_lane_bands_2_10_mre_pct"],
        "next_gate": "10k_head_selection",
        "finalizer_recovery": True,
    })
    write_json(out / "finalizer_recovery.json", {
        "status": "recovered",
        "reason": "original finalizer JSON circular reference",
        "source_slurm_job_id": source_slurm_job_id,
        "source_history_csv": str(out / "history.csv"),
        "source_checkpoint_sha256": _sha256_checkpoint(checkpoint),
        "source_terminal_checkpoint_sha256": metrics["terminal_checkpoint_sha256"],
        "recovery_scope": "missing finalizer artifacts only; no training or checkpoint rewrite",
    })
    missing = validate_run_artifacts(out)
    if missing:
        raise RuntimeError(f"incomplete recovered dual-lane artifact contract: {missing}")
    return out


__all__ = ["recover_fixed_200_finalizer"]
