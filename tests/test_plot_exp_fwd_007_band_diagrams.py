"""Tests for EXP-FWD-007 representative band-diagram selection.

Author: fangtoast
Email: fangtoast@foxmail.com
Copyright (c) 2026 fangtoast. All rights reserved.
"""

from __future__ import annotations

import numpy as np

from scripts import plot_exp_fwd_007_band_diagrams as plotter


def test_complete_gaps_use_full_bz_extrema() -> None:
    target = np.asarray(
        [
            [[0.10, 0.30, 0.50], [0.20, 0.32, 0.56]],
            [[0.15, 0.31, 0.52], [0.18, 0.33, 0.58]],
        ]
    ).reshape(4, 3)

    gaps = plotter._complete_gaps(target)

    assert gaps == [(1, 0.2, 0.3), (2, 0.33, 0.5)]


def test_gap_sample_selection_returns_low_median_and_high_error() -> None:
    target = np.zeros((5, 2, 3), dtype=float)
    target[:, :, 0] = 0.1
    target[:, :, 1] = 0.2
    target[:, :, 2] = 0.3
    target[1, 1, 0] = 0.25
    target[1, 1, 1] = 0.35
    target[3, 1, 0] = 0.25
    target[3, 1, 1] = 0.35
    sample_mre = np.asarray([0.7, 0.1, 0.5, 0.2, 0.9])

    selected = plotter._select_gap_samples(target, sample_mre)

    assert selected == [2, 0, 4]
