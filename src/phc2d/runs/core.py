"""Shared run-directory and logging helpers.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping

RUN_LOG_NAME = "run.log"
RUN_META_NAME = "run.meta.json"
RUN_RESULTS_NAME = "results.json"
RUN_METRICS_NAME = "metrics.json"
RUN_RESOLVED_CONFIG_NAME = "config.resolved.yaml"
RUN_FIGURES_DIR = "figures"


@contextmanager
def measure_wall_time(metrics: MutableMapping[str, Any], key: str) -> Iterator[None]:
    """Measure wall-clock seconds into a metrics mapping."""

    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = float(time.perf_counter() - start)


def add_throughput_metrics(
    metrics: MutableMapping[str, Any],
    *,
    count: int,
    elapsed_s: float,
    prefix: str = "samples",
) -> MutableMapping[str, Any]:
    """Add per-item latency and throughput metrics for a measured block."""

    n = max(int(count), 0)
    elapsed = max(float(elapsed_s), 0.0)
    singular = prefix[:-1] if prefix.endswith("s") else prefix
    metrics[f"{prefix}_per_s"] = float(n / elapsed) if n > 0 and elapsed > 0.0 else 0.0
    metrics[f"time_per_{singular}_ms"] = float(1000.0 * elapsed / n) if n > 0 else 0.0
    return metrics


def make_run_dir(
    task: str,
    *,
    runs_dir: str | Path = "runs",
    run_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Create or resolve ``runs_dir/{yyyymmdd}/{task}_{YYYYMMDDHHmm}/``."""
    if run_dir is not None:
        out = Path(run_dir)
    else:
        ts = timestamp or datetime.now()
        out = Path(runs_dir) / ts.strftime("%Y%m%d") / f"{task}_{ts.strftime('%Y%m%d%H%M')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_run_dir(runs_root: str | Path, name: str, run_dir: str | Path | None = None) -> Path:
    """Legacy-compatible resolver with explicit ``runs_root`` then task name."""
    return make_run_dir(name, runs_dir=runs_root, run_dir=run_dir)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write pretty JSON and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def write_results(run_dir: str | Path, results: Mapping[str, Any]) -> Path:
    """Write ``results.json`` under a run directory."""
    return write_json(Path(run_dir) / RUN_RESULTS_NAME, results)


def write_metrics(run_dir: str | Path, metrics: Mapping[str, Any]) -> Path:
    """Write ``metrics.json`` under a run directory."""
    return write_json(Path(run_dir) / RUN_METRICS_NAME, metrics)


def write_resolved_config(run_dir: str | Path, config_yaml: str) -> Path:
    """Write the resolved YAML config snapshot under a run directory."""
    out = Path(run_dir) / RUN_RESOLVED_CONFIG_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(str(config_yaml), encoding="utf-8")
    return out


def ensure_figures_dir(run_dir: str | Path) -> Path:
    """Create and return the standard figures directory."""
    out = Path(run_dir) / RUN_FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_run_meta(run_dir: str | Path, meta: Mapping[str, Any]) -> Path:
    """Write ``run.meta.json`` with UTC timestamp and current git commit when available."""
    payload = dict(meta)
    payload["utc"] = datetime.now(timezone.utc).isoformat()
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        payload["git_commit"] = os.environ.get("PHC2D_DEPLOYED_GIT_COMMIT") or None
    return write_json(Path(run_dir) / RUN_META_NAME, payload)


def validate_run_artifacts(run_dir: str | Path) -> list[str]:
    """Validate required run artifacts and return missing relative paths.

    Parameters
    ----------
    run_dir:
        Candidate run directory.
    Returns
    -------
    list[str]
        A list of missing required artifact names. Empty means pass.
    """
    root = Path(run_dir)
    required_file_names = [
        RUN_LOG_NAME,
        RUN_META_NAME,
        RUN_RESOLVED_CONFIG_NAME,
        "results.json",
        RUN_METRICS_NAME,
        "history.csv",
    ]
    missing: list[str] = []
    if not root.is_dir():
        return ["run_dir"]
    for name in required_file_names:
        if not (root / name).is_file():
            missing.append(name)
    if not (root / RUN_FIGURES_DIR).is_dir():
        missing.append(f"{RUN_FIGURES_DIR}/")
    return missing


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    *,
    run_meta: Mapping[str, Any] | None = None,
) -> logging.Logger:
    """Configure root logging and optionally write run metadata."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(formatter)
        root.addHandler(handler)
        if run_meta is not None:
            write_run_meta(log_path.parent, run_meta)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
