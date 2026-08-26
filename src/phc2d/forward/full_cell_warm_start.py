"""Weight-only warm-start contracts for full-cell convergence continuations.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_history(path: Path) -> list[dict[str, Any]]:
    integer_fields = {"epoch", "optimizer_steps", "samples_seen"}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: int(value) if key in integer_fields else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def load_weight_only_warm_start(
    source_run: str | Path,
    *,
    expected_model: str,
    expected_seed: int,
    torch_module: Any,
    expected_source_experiment: str | None = None,
) -> dict[str, Any]:
    """Validate and load an epoch-200 source without claiming optimizer resume."""
    source = Path(source_run).resolve()
    required = ("history.csv", "results.json", "config.resolved.yaml", "model.pt", "epoch200.pt")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"warm-start source artifacts missing: {missing}")
    result = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if str(result.get("model")) != expected_model or int(result.get("seed", -1)) != int(expected_seed):
        raise RuntimeError("warm-start source model/seed identity mismatch")
    if int(result.get("stop_epoch", -1)) != 200 or int(result.get("history_epoch_count", -1)) != 200:
        raise RuntimeError("warm-start source must be a complete epoch-200 run")
    source_config = yaml.safe_load(
        (source / "config.resolved.yaml").read_text(encoding="utf-8")
    )
    source_experiment = str((source_config or {}).get("experiment", {}).get("experiment_id", ""))
    if expected_source_experiment is not None and source_experiment != expected_source_experiment:
        raise RuntimeError("warm-start source experiment identity mismatch")
    history = _read_history(source / "history.csv")
    if [row["epoch"] for row in history] != list(range(1, 201)):
        raise RuntimeError("warm-start source history must cover exactly epochs 1-200")
    model_sha = _sha256(source / "model.pt")
    terminal_sha = _sha256(source / "epoch200.pt")
    if result.get("checkpoint_sha256") != model_sha or result.get("terminal_checkpoint_sha256") != terminal_sha:
        raise RuntimeError("warm-start source checkpoint hash mismatch")
    try:
        terminal = torch_module.load(source / "epoch200.pt", map_location="cpu", weights_only=True)
    except TypeError:
        terminal = torch_module.load(source / "epoch200.pt", map_location="cpu")
    if int(terminal.get("epoch", -1)) != 200 or "state_dict" not in terminal:
        raise RuntimeError("warm-start terminal checkpoint is not an epoch-200 state")
    return {
        "source_run": str(source),
        "source_history": history,
        "source_result": result,
        "terminal_payload": terminal,
        "source_best_checkpoint": str(source / "model.pt"),
        "source_best_checkpoint_sha256": model_sha,
        "source_terminal_checkpoint": str(source / "epoch200.pt"),
        "source_terminal_checkpoint_sha256": terminal_sha,
        "source_config_sha256": _sha256(source / "config.resolved.yaml"),
        "source_experiment_id": source_experiment,
        "source_epoch": 200,
        "optimizer_state_restored": False,
        "optimizer_reset_reason": "source epoch200.pt contains weights but no AdamW state",
        "data_order_state_restored": False,
    }


def seed_derivative_best_checkpoint(warm_start: Mapping[str, Any], output: str | Path) -> Path:
    """Seed derivative selection with the unchanged source best checkpoint."""
    destination = Path(output) / "model.pt"
    shutil.copy2(Path(str(warm_start["source_best_checkpoint"])), destination)
    if _sha256(destination) != str(warm_start["source_best_checkpoint_sha256"]):
        raise RuntimeError("copied source best checkpoint hash mismatch")
    return destination


def rolling_convergence_record(
    history: Sequence[Mapping[str, Any]],
    *,
    end_epoch: int,
    metric: str,
    threshold: float,
) -> dict[str, Any]:
    """Compare the two ten-epoch windows ending at ``end_epoch``."""
    first = (int(end_epoch) - 19, int(end_epoch) - 10)
    second = (int(end_epoch) - 9, int(end_epoch))
    by_epoch = {int(row["epoch"]): row for row in history}
    epochs = list(range(first[0], second[1] + 1))
    missing = [epoch for epoch in epochs if epoch not in by_epoch]
    if missing:
        raise ValueError(f"rolling convergence history missing epochs: {missing}")
    values = [float(by_epoch[epoch][metric]) for epoch in epochs]
    import math

    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"rolling convergence metric {metric} contains non-finite values")
    mean_first = sum(values[:10]) / 10.0
    mean_second = sum(values[10:]) / 10.0
    relative_change = abs(mean_second - mean_first) / max(abs(mean_first), 1.0e-12)
    return {
        "metric": metric,
        "first_window": list(first),
        "second_window": list(second),
        "mean_first_window": mean_first,
        "mean_second_window": mean_second,
        "relative_change": relative_change,
        "threshold": float(threshold),
        "passed": bool(relative_change <= float(threshold)),
        "checked_at_epoch": int(end_epoch),
    }


def public_warm_start_metadata(warm_start: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe provenance without loaded tensors or source history rows."""
    excluded = {"source_history", "source_result", "terminal_payload"}
    return {key: value for key, value in warm_start.items() if key not in excluded}


__all__ = [
    "load_weight_only_warm_start",
    "public_warm_start_metadata",
    "rolling_convergence_record",
    "seed_derivative_best_checkpoint",
]
