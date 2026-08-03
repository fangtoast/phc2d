"""Matplotlib plotting primitives shared by legacy and AI4PhC scripts.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from matplotlib.colors import ListedColormap

from phc2d.data import epsilon_from_rho, lift_rho_to_grid

_BINARY_CMAP = ListedColormap(["white", "#173f7a"])


@dataclass(frozen=True)
class PlotRequest:
    """Generic plot request used by thin CLI wrappers."""

    output: Path
    title: str = ""
    dpi: int = 180
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlotResult:
    """Path and metadata for a saved figure."""

    path: Path
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_square_grid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr
    flat = arr.reshape(-1)
    if flat.size == 36:
        return lift_rho_to_grid(flat)
    side = int(np.sqrt(flat.size))
    if side * side != flat.size:
        raise ValueError(f"Cannot reshape array of size {flat.size} to a square grid.")
    return flat.reshape(side, side)


def plot_medium_map(
    rho: np.ndarray,
    output: str | Path,
    *,
    mode: str = "binary",
    title: str = "",
    threshold: float = 0.5,
    eps_air: float = 1.0,
    eps_alum: float = 8.9,
    show_grid: bool = False,
    show_axes: bool = False,
    dpi: int = 180,
) -> PlotResult:
    """Save one binary or epsilon medium map."""
    import matplotlib.pyplot as plt

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    if mode == "binary":
        img = (_as_square_grid(rho) >= threshold).astype(np.int8)
        ax.imshow(img, cmap=_BINARY_CMAP, origin="lower", interpolation="nearest", vmin=0, vmax=1)
    elif mode == "epsilon":
        img = epsilon_from_rho(np.asarray(rho), eps_air=eps_air, eps_alum=eps_alum)
        im = ax.imshow(img, cmap="Blues", origin="lower", interpolation="nearest", vmin=eps_air, vmax=eps_alum)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Relative permittivity")
    else:
        raise ValueError(f"Unknown medium plot mode: {mode}")
    if show_grid:
        n = int(img.shape[0])
        ax.set_xticks(np.arange(-0.5, n, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1.0), minor=True)
        ax.grid(which="minor", color="#d9d9d9", linewidth=0.5)
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return PlotResult(path=out, kind="medium_map", metadata={"mode": mode})


def plot_band_overlay(
    target: np.ndarray,
    output: str | Path,
    *,
    prediction: np.ndarray | None = None,
    title: str = "Band overlay",
    dpi: int = 180,
) -> PlotResult:
    """Save target and optional prediction band curves over k index."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(target, dtype=np.float64)
    if y_true.ndim != 2:
        raise ValueError(f"Expected target shape (nk, nb), got {y_true.shape}.")
    y_pred = None if prediction is None else np.asarray(prediction, dtype=np.float64)
    x = np.arange(y_true.shape[0])
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for band in range(y_true.shape[1]):
        label = "target" if band == 0 else None
        ax.plot(x, y_true[:, band], color="black", alpha=0.55, linewidth=1.0, label=label)
        if y_pred is not None:
            pred_label = "prediction" if band == 0 else None
            ax.plot(x, y_pred[:, band], color="#d95f02", alpha=0.8, linewidth=1.0, linestyle="--", label=pred_label)
    ax.set_xlabel("k index")
    ax.set_ylabel("normalized frequency")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return PlotResult(path=out, kind="band_overlay", metadata={"has_prediction": y_pred is not None})


def plot_metric_report(
    metrics: Mapping[str, Any],
    output: str | Path,
    *,
    title: str = "Metric report",
    dpi: int = 180,
) -> PlotResult:
    """Save a compact horizontal bar chart for scalar metrics."""
    import matplotlib.pyplot as plt

    scalars: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            scalars[str(key)] = float(value)
    if not scalars:
        raise ValueError("No scalar numeric metrics available to plot.")
    items = list(scalars.items())[:16]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.34 * len(items))))
    ax.barh(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return PlotResult(path=out, kind="metric_report", metadata={"n_metrics": len(items)})


def plot_field_map(
    field: np.ndarray,
    output: str | Path,
    *,
    title: str = "Field map",
    cmap: str = "magma",
    dpi: int = 180,
) -> PlotResult:
    """Save a scalar field map."""
    import matplotlib.pyplot as plt

    arr = np.squeeze(np.asarray(field, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D scalar field, got {arr.shape}.")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(arr, origin="lower", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return PlotResult(path=out, kind="field_map", metadata={"shape": list(arr.shape)})
