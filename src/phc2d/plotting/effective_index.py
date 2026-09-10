"""Plots for effective-index and folded-gap audits.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def write_effective_index_figures(
    figures_dir: Path,
    slope_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[str],
    lanes: Sequence[str],
) -> None:
    """Write compact slope and gap-alignment diagnostic figures."""

    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    if slope_rows:
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        for lane, marker in zip(lanes, ("o", "s"), strict=False):
            selected = [row for row in slope_rows if row["lane"] == lane]
            for field, label in (("old_index", "old"), ("te_index", "TE")):
                axis.scatter(
                    [row[field] for row in selected],
                    [row["oracle_index_mean"] for row in selected],
                    s=14,
                    alpha=0.7,
                    marker=marker,
                    label=f"{lane}: {label}",
                )
        axis.set_xlabel("analytic effective index")
        axis.set_ylabel("MPB slope effective index")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(figures_dir / "slope_fit_effective_index.png", dpi=160)
        plt.close(figure)
    if gap_rows:
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        values = [
            float(
                np.mean(
                    [row["distance_improvement"] for row in gap_rows if row["candidate"] == name]
                )
            )
            for name in candidates
        ]
        axis.bar(list(candidates), values)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel("old - TE normalized center distance")
        figure.tight_layout()
        figure.savefig(figures_dir / "gap_alignment_improvement.png", dpi=160)
        plt.close(figure)


__all__ = ["write_effective_index_figures"]
