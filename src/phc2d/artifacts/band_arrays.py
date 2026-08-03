"""Helpers for stable band-array artifact names.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

TARGET_BANDS_NAME = "target_bands.npy"
SURROGATE_PRED_BANDS_NAME = "surrogate_pred_bands.npy"
MPB_REPLAY_BANDS_NAME = "mpb_replay_bands.npy"


def _save_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return {
        "path": path.name,
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
    }


def save_forward_band_arrays(
    run_dir: Path,
    *,
    target_bands: np.ndarray,
    surrogate_pred_bands: np.ndarray,
) -> dict[str, Any]:
    """Save forward target and surrogate predictions with fixed filenames."""

    return {
        "target_bands": _save_array(run_dir / TARGET_BANDS_NAME, target_bands),
        "surrogate_pred_bands": _save_array(run_dir / SURROGATE_PRED_BANDS_NAME, surrogate_pred_bands),
        "mpb_replay_bands": None,
        "mpb_replay_available": False,
        "note": "forward runs do not perform MPB replay; no mpb_replay_bands.npy is written.",
    }


def save_inverse_replay_band_arrays(
    run_dir: Path,
    *,
    target_bands: np.ndarray,
    surrogate_pred_bands: np.ndarray,
    mpb_replay_bands: np.ndarray,
    surrogate_available_count: int,
    surrogate_sources: list[str],
) -> dict[str, Any]:
    """Save inverse replay target, surrogate, and MPB replay bands."""

    target = np.asarray(target_bands, dtype=np.float32)
    surrogate = np.asarray(surrogate_pred_bands, dtype=np.float32)
    replay = np.asarray(mpb_replay_bands, dtype=np.float32)
    if target.shape != surrogate.shape or target.shape != replay.shape:
        raise ValueError(
            "Band arrays must have matching shapes: "
            f"target={target.shape}, surrogate={surrogate.shape}, replay={replay.shape}"
        )
    return {
        "target_bands": _save_array(run_dir / TARGET_BANDS_NAME, target),
        "surrogate_pred_bands": _save_array(run_dir / SURROGATE_PRED_BANDS_NAME, surrogate),
        "mpb_replay_bands": _save_array(run_dir / MPB_REPLAY_BANDS_NAME, replay),
        "surrogate_pred_available_count": int(surrogate_available_count),
        "surrogate_pred_missing_count": int(target.shape[0] - surrogate_available_count),
        "surrogate_sources": sorted({str(source) for source in surrogate_sources}),
        "missing_surrogate_values": "nan_placeholder",
    }
