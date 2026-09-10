"""Shared evidence and statistics support for EXP-PHY-004.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
SOURCES = ("real_mpb", "convnext", "joint")


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path without changing absolute paths."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative_path(path: Path) -> str:
    """Return a portable repository-relative path when possible."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def decode_text(value: Any) -> str:
    """Decode HDF5 byte strings consistently."""

    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def jsonable(value: Any) -> Any:
    """Convert path and NumPy values into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of an evidence file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV as dictionaries."""

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write non-empty dictionary rows with a stable discovered schema."""

    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_surrogate_physics_config(path: Path) -> dict[str, Any]:
    """Load and validate the top-level EXP-PHY-004 contract."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("surrogate-physics config must be a mapping")
    config = dict(raw)
    if str(config.get("experiment", {}).get("experiment_id")) != "EXP-PHY-004":
        raise ValueError("surrogate-physics config must use experiment_id EXP-PHY-004")
    if set(config.get("datasets", {})) != {"B", "G"}:
        raise ValueError("surrogate-physics config must define B and G datasets")
    formats = tuple(str(value) for value in config.get("output", {}).get("formats", ()))
    if "svg" not in formats or any(value not in {"png", "svg", "pdf", "tiff"} for value in formats):
        raise ValueError("output formats must include svg and use png/svg/pdf/tiff only")
    return config


def select_quantile_positions(values: np.ndarray, quantiles: Sequence[float]) -> list[tuple[float, int, float]]:
    """Select deterministic positions nearest configured empirical quantiles."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < len(quantiles) or np.any(~np.isfinite(array)):
        raise ValueError("quantile selection requires enough finite values")
    selected: list[tuple[float, int, float]] = []
    used: set[int] = set()
    for raw_quantile in quantiles:
        quantile = float(raw_quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError("selection quantiles must lie in (0, 1)")
        target = float(np.quantile(array, quantile))
        order = np.lexsort((np.arange(array.size), np.abs(array - target)))
        position = next(int(item) for item in order if int(item) not in used)
        used.add(position)
        selected.append((quantile, position, target))
    return selected


def quantile_role_label(quantile: float) -> str:
    """Return a reader-facing difficulty role for an empirical quantile."""

    value = float(quantile)
    if not 0.0 < value < 1.0:
        raise ValueError("selection quantiles must lie in (0, 1)")
    if np.isclose(value, 0.5):
        return "median"
    return "low-error" if value < 0.5 else "high-error"


def sample_mre_pct(prediction: np.ndarray, target: np.ndarray, band_slice: slice = slice(1, 10)) -> np.ndarray:
    """Return per-sample MRE in percent for the configured band slice."""

    relative = np.abs(np.asarray(prediction)[:, :, band_slice] - np.asarray(target)[:, :, band_slice]) / np.maximum(
        np.abs(np.asarray(target)[:, :, band_slice]), 1.0e-8
    )
    return 100.0 * np.mean(relative, axis=(1, 2))


def paired_bootstrap_mean_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 250,
) -> dict[str, float]:
    """Return a deterministic paired bootstrap for mean ``left - right``."""

    difference = np.asarray(left, dtype=np.float64).reshape(-1) - np.asarray(right, dtype=np.float64).reshape(-1)
    if difference.size == 0 or np.any(~np.isfinite(difference)):
        raise ValueError("paired bootstrap requires finite paired values")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        count = min(int(chunk_size), int(replicates) - start)
        indices = rng.integers(0, difference.size, size=(count, difference.size))
        estimates[start : start + count] = np.mean(difference[indices], axis=1)
    return {
        "mean_difference_pct_points": float(np.mean(difference)),
        "ci_low_pct_points": float(np.percentile(estimates, 2.5)),
        "ci_high_pct_points": float(np.percentile(estimates, 97.5)),
    }


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the phase-level history table."""

    write_csv_rows(path, rows)


__all__ = [
    "ROOT",
    "SOURCES",
    "decode_text",
    "hash_file",
    "jsonable",
    "load_surrogate_physics_config",
    "paired_bootstrap_mean_difference",
    "quantile_role_label",
    "read_csv_rows",
    "relative_path",
    "repo_path",
    "sample_mre_pct",
    "select_quantile_positions",
    "write_csv_rows",
    "write_history",
]
